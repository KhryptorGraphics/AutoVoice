# AutoVoice Model Architecture

Technical deep dive into the So-VITS-SVC singing voice conversion architecture and implementation.

## 1. Overview

### System Architecture

AutoVoice implements So-VITS-SVC (Soft-VC-based Singing Voice Conversion) with GPU acceleration and production optimizations.

**High-Level Pipeline**:
```
Input Song → Vocal Separation → Pitch Extraction → Voice Conversion → Audio Synthesis → Output
```

**Core Components**:
1. **VocalSeparator**: Demucs-based source separation
2. **SingingPitchExtractor**: Torchcrepe F0 tracking with vibrato analysis
3. **SingingVoiceConverter**: So-VITS-SVC model with ContentEncoder, PitchEncoder, FlowDecoder
4. **HiFiGAN Vocoder**: Neural audio synthesis
5. **SpeakerEncoder**: Resemblyzer-based speaker embeddings

### Design Philosophy

**Separation of Concerns**:
- Clear module boundaries for testability
- Independent optimization of each component
- Modular architecture for easy extension

**Performance First**:
- GPU acceleration throughout pipeline
- TensorRT optimization for inference
- Intelligent caching strategies
- Batch processing support

**Quality Assurance**:
- Comprehensive quality metrics
- Validation at each pipeline stage
- Robust error handling and recovery

## 2. Component Architecture

### 2.1 VocalSeparator

**Location**: `src/auto_voice/audio/source_separator.py`

**Purpose**: Separate vocals from instrumental using state-of-the-art source separation models.

#### Architecture

```python
class VocalSeparator:
    def __init__(
        self,
        model_name: str = 'htdemucs',
        device: str = 'cuda',
        cache_dir: Optional[str] = None
    )
```

**Supported Models**:
- **htdemucs** (default): Hybrid Transformer Demucs, 4-stem separation (vocals, drums, bass, other)
- **htdemucs_ft**: Fine-tuned variant for better vocal quality
- **mdx_extra**: MDX-Net model for high-quality separation
- **spleeter**: Faster model for real-time applications

**Implementation Details**:

**Model Loading**:
```python
def _load_model(self) -> None:
    """Load Demucs model with GPU optimization"""
    self.separator = get_model(self.model_name)
    self.separator.to(self.device)
    self.separator.eval()
```

**Separation Pipeline**:
```python
def separate(
    self,
    audio_path: str,
    output_dir: str,
    use_cache: bool = True
) -> Dict[str, str]:
    """
    Separate vocals from instrumental

    Pipeline:
    1. Load audio and resample to model rate (44.1kHz)
    2. Apply model-specific preprocessing
    3. Run separation model with TTA (Test-Time Augmentation)
    4. Post-process and normalize outputs
    5. Save separated stems
    """
    # Load and preprocess
    audio = self._load_audio(audio_path)
    audio = self._preprocess(audio)

    # Separate with TTA for quality
    sources = apply_model(
        self.separator,
        audio,
        shifts=2,  # TTA shifts for robustness
        split=True,  # Split processing for memory
        overlap=0.25  # Overlap for seamless stitching
    )

    # Extract vocals and instrumental
    vocals = sources[self.sources.index('vocals')]
    instrumental = sum([
        sources[i] for i in range(len(sources))
        if i != self.sources.index('vocals')
    ])

    return {
        'vocals': vocals_path,
        'instrumental': instrumental_path
    }
```

**Caching Strategy**:
```python
def _get_cache_key(self, audio_path: str) -> str:
    """Generate cache key from audio hash"""
    with open(audio_path, 'rb') as f:
        return hashlib.sha256(f.read()).hexdigest()
```

**Memory Optimization**:
- Split processing for long audio (>5 minutes)
- Automatic batch size adjustment based on GPU memory
- Gradient checkpointing disabled in inference mode

**Quality Metrics**:
- SDR (Source-to-Distortion Ratio): >10 dB
- SIR (Source-to-Interference Ratio): >15 dB
- SAR (Source-to-Artifacts Ratio): >10 dB

### 2.2 SingingPitchExtractor

**Location**: `src/auto_voice/audio/pitch_extractor.py`

**Purpose**: Extract fundamental frequency (F0) contour with singing-specific enhancements.

#### Architecture

```python
class SingingPitchExtractor:
    def __init__(
        self,
        model_name: str = 'full',
        device: str = 'cuda',
        hop_length: int = 512,
        sample_rate: int = 44100
    )
```

**Features**:
- **Torchcrepe-based F0 tracking**: State-of-the-art pitch detection
- **Vibrato detection**: Identify and preserve singing vibrato
- **Pitch smoothing**: Remove jitter while preserving expression
- **Confidence scoring**: Filter low-confidence pitch estimates

#### Implementation Details

**F0 Extraction Pipeline**:
```python
def extract_f0(
    self,
    audio: np.ndarray,
    sample_rate: int = 44100,
    return_confidence: bool = True
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Extract F0 contour using torchcrepe

    Pipeline:
    1. Resample to torchcrepe expected rate (16kHz)
    2. Apply CREPE model for pitch tracking
    3. Smooth F0 with median filtering
    4. Detect and preserve vibrato
    5. Interpolate unvoiced regions
    """
    # Resample for CREPE
    if sample_rate != 16000:
        audio_resampled = librosa.resample(
            audio, orig_sr=sample_rate, target_sr=16000
        )

    # Extract pitch with torchcrepe
    f0, confidence = torchcrepe.predict(
        audio_resampled,
        sample_rate=16000,
        hop_length=self.hop_length,
        fmin=50,  # Lowest expected pitch
        fmax=800,  # Highest expected pitch
        model=self.model_name,
        batch_size=512,
        device=self.device,
        return_periodicity=True
    )

    # Smooth F0 while preserving vibrato
    f0 = self._smooth_f0(f0, confidence)

    # Detect vibrato regions
    vibrato_mask = self._detect_vibrato(f0)

    return f0, confidence
```

**Vibrato Detection**:
```python
def _detect_vibrato(
    self,
    f0: np.ndarray,
    rate_threshold: float = 5.0,  # Hz
    depth_threshold: float = 0.02  # 2% of F0
) -> np.ndarray:
    """
    Detect vibrato using spectral analysis

    Vibrato characteristics:
    - Rate: 5-7 Hz modulation
    - Depth: 1-3% of fundamental frequency
    - Regular periodic pattern
    """
    # Compute F0 modulation spectrum
    f0_spectrum = np.fft.fft(f0)
    freqs = np.fft.fftfreq(len(f0), d=self.hop_length/self.sample_rate)

    # Find peaks in 5-7 Hz range
    vibrato_range = (freqs >= 5.0) & (freqs <= 7.0)
    vibrato_energy = np.abs(f0_spectrum[vibrato_range])

    # Threshold for vibrato detection
    vibrato_mask = vibrato_energy > depth_threshold * np.mean(f0)

    return vibrato_mask
```

**F0 Smoothing**:
```python
def _smooth_f0(
    self,
    f0: np.ndarray,
    confidence: np.ndarray,
    threshold: float = 0.8
) -> np.ndarray:
    """
    Smooth F0 contour while preserving expression

    Strategy:
    - Median filter for jitter removal
    - Preserve high-confidence rapid changes (vibrato, pitch bends)
    - Interpolate low-confidence regions
    """
    # Filter low confidence
    f0_filtered = f0.copy()
    f0_filtered[confidence < threshold] = 0

    # Median filter for smoothing
    from scipy.signal import medfilt
    f0_smooth = medfilt(f0_filtered, kernel_size=5)

    # Preserve vibrato regions
    vibrato_mask = self._detect_vibrato(f0)
    f0_smooth[vibrato_mask] = f0_filtered[vibrato_mask]

    return f0_smooth
```

**CUDA Acceleration**:
```python
# Custom CUDA kernel for F0 processing
from auto_voice.cuda import launch_pitch_detection

f0_cuda = launch_pitch_detection(
    audio_tensor,
    sample_rate=44100,
    hop_length=512,
    fmin=50.0,
    fmax=800.0
)
```

### 2.3 SingingVoiceConverter

**Location**: `src/auto_voice/models/singing_voice_converter.py`

**Purpose**: Core So-VITS-SVC model for voice conversion.

#### Architecture

The So-VITS-SVC model consists of four main components:

```
Input Audio → ContentEncoder → [content_features]
                                       ↓
Target Speaker → SpeakerEncoder → [speaker_embedding]
                                       ↓
Pitch Contour → PitchEncoder → [pitch_features]
                                       ↓
                    FlowDecoder([content, speaker, pitch])
                                       ↓
                    HiFiGAN Vocoder
                                       ↓
                    Output Audio
```

#### 2.3.1 ContentEncoder

**Location**: `src/auto_voice/models/content_encoder.py`

**Purpose**: Extract content features independent of speaker identity.

**Architecture**:
```python
class ContentEncoder(nn.Module):
    def __init__(
        self,
        in_channels: int = 256,
        hidden_channels: int = 192,
        filter_channels: int = 768,
        n_layers: int = 6,
        n_heads: int = 2,
        kernel_size: int = 3
    ):
        super().__init__()

        # Prenet for input processing
        self.prenet = nn.Sequential(
            nn.Conv1d(in_channels, hidden_channels, 5, padding=2),
            nn.ReLU(),
            nn.Dropout(0.1)
        )

        # Multi-head self-attention layers
        self.encoder_layers = nn.ModuleList([
            TransformerEncoderLayer(
                hidden_channels,
                n_heads,
                filter_channels,
                kernel_size,
                dropout=0.1
            )
            for _ in range(n_layers)
        ])

        # Output projection
        self.projection = nn.Conv1d(hidden_channels, hidden_channels, 1)
```

**Forward Pass**:
```python
def forward(
    self,
    x: torch.Tensor,
    x_mask: torch.Tensor
) -> torch.Tensor:
    """
    Extract content features

    Args:
        x: Input features [B, C, T]
        x_mask: Padding mask [B, 1, T]

    Returns:
        Content features [B, C, T]
    """
    # Prenet processing
    x = self.prenet(x) * x_mask

    # Self-attention layers
    for layer in self.encoder_layers:
        x = layer(x, x_mask)

    # Output projection
    x = self.projection(x) * x_mask

    return x
```

**Key Features**:
- Multi-head self-attention for long-range dependencies
- Residual connections for gradient flow
- Layer normalization for stable training
- Dropout for regularization

#### 2.3.2 SpeakerEncoder

**Location**: `src/auto_voice/models/speaker_encoder.py`

**Purpose**: Extract speaker embedding from reference audio.

**Architecture**:
```python
class SpeakerEncoder(nn.Module):
    def __init__(
        self,
        mel_n_channels: int = 80,
        model_embedding_size: int = 256,
        model_hidden_size: int = 256,
        model_num_layers: int = 3
    ):
        super().__init__()

        # LSTM for temporal modeling
        self.lstm = nn.LSTM(
            mel_n_channels,
            model_hidden_size,
            model_num_layers,
            batch_first=True
        )

        # Linear projection to embedding
        self.linear = nn.Linear(model_hidden_size, model_embedding_size)

        # ReLU activation
        self.relu = nn.ReLU()
```

**Embedding Extraction**:
```python
def forward(self, mel: torch.Tensor) -> torch.Tensor:
    """
    Extract speaker embedding from mel spectrogram

    Args:
        mel: Mel spectrogram [B, T, n_mels]

    Returns:
        Speaker embedding [B, embedding_size]
    """
    # LSTM processing
    _, (hidden, _) = self.lstm(mel)

    # Take last hidden state
    embedding = hidden[-1]

    # Project to embedding space
    embedding = self.linear(embedding)
    embedding = self.relu(embedding)

    # L2 normalize for cosine similarity
    embedding = F.normalize(embedding, p=2, dim=1)

    return embedding
```

**Training Strategy**:
- GE2E (Generalized End-to-End) loss for speaker discrimination
- Hard negative mining for robust embeddings
- Data augmentation: speed perturbation, SpecAugment

**Speaker Similarity**:
```python
def compute_similarity(
    self,
    embedding1: torch.Tensor,
    embedding2: torch.Tensor
) -> torch.Tensor:
    """Cosine similarity between speaker embeddings"""
    return F.cosine_similarity(embedding1, embedding2, dim=1)
```

#### 2.3.3 PitchEncoder

**Location**: `src/auto_voice/models/pitch_encoder.py`

**Purpose**: Encode pitch contour for expressive control.

**Architecture**:
```python
class PitchEncoder(nn.Module):
    def __init__(
        self,
        n_vocab: int = 360,  # F0 range in semitones (30Hz-8kHz)
        out_channels: int = 192,
        hidden_channels: int = 192,
        filter_channels: int = 768,
        n_layers: int = 3,
        kernel_size: int = 3
    ):
        super().__init__()

        # Pitch embedding
        self.pitch_embedding = nn.Embedding(n_vocab, hidden_channels)

        # Convolutional layers for pitch processing
        self.conv_layers = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(
                    hidden_channels if i == 0 else filter_channels,
                    filter_channels,
                    kernel_size,
                    padding=kernel_size//2
                ),
                nn.ReLU(),
                nn.LayerNorm(filter_channels),
                nn.Dropout(0.1)
            )
            for i in range(n_layers)
        ])

        # Output projection
        self.projection = nn.Conv1d(filter_channels, out_channels, 1)
```

**Forward Pass**:
```python
def forward(
    self,
    f0: torch.Tensor,
    f0_mask: torch.Tensor
) -> torch.Tensor:
    """
    Encode pitch contour

    Args:
        f0: F0 sequence in Hz [B, T]
        f0_mask: Padding mask [B, 1, T]

    Returns:
        Pitch features [B, C, T]
    """
    # Convert Hz to semitone bins
    f0_semitones = self._hz_to_semitones(f0)
    f0_bins = torch.clamp(f0_semitones, 0, self.n_vocab - 1).long()

    # Embed pitch
    x = self.pitch_embedding(f0_bins).transpose(1, 2)

    # Convolutional processing
    for conv in self.conv_layers:
        x = conv(x * f0_mask)

    # Output projection
    x = self.projection(x) * f0_mask

    return x

def _hz_to_semitones(self, f0: torch.Tensor) -> torch.Tensor:
    """Convert Hz to semitone bins relative to A4 (440Hz)"""
    # f0 = 0 for unvoiced
    f0_semitones = torch.zeros_like(f0)
    voiced_mask = f0 > 0
    f0_semitones[voiced_mask] = 12 * torch.log2(f0[voiced_mask] / 440.0) + 69
    return f0_semitones
```

**Pitch Representation**:
- Logarithmic scale (semitones) for perceptual uniformity
- Embedding lookup for discrete pitch values
- Continuous interpolation for smooth transitions

#### 2.3.4 FlowDecoder

**Location**: `src/auto_voice/models/flow_decoder.py`

**Purpose**: Transform content features to target speaker using normalizing flows.

**Architecture**:
```python
class FlowDecoder(nn.Module):
    def __init__(
        self,
        in_channels: int = 192,
        hidden_channels: int = 192,
        kernel_size: int = 5,
        dilation_rate: int = 1,
        n_blocks: int = 4,
        n_layers: int = 4,
        n_flows: int = 4,
        gin_channels: int = 256
    ):
        super().__init__()

        # Pre-flow
        self.pre_flow = nn.Sequential(
            nn.Conv1d(in_channels, hidden_channels, 1),
            nn.ReLU()
        )

        # Normalizing flows
        self.flows = nn.ModuleList()
        for i in range(n_flows):
            self.flows.append(
                ResidualCouplingBlock(
                    hidden_channels,
                    hidden_channels,
                    kernel_size,
                    dilation_rate,
                    n_layers,
                    gin_channels=gin_channels,
                    mean_only=True
                )
            )
            self.flows.append(Flip())

        # Post-flow
        self.post_flow = nn.Conv1d(hidden_channels, in_channels, 1)
```

**Forward Pass (Inference)**:
```python
def forward(
    self,
    content: torch.Tensor,
    pitch: torch.Tensor,
    speaker: torch.Tensor,
    mask: torch.Tensor,
    temperature: float = 1.0
) -> torch.Tensor:
    """
    Generate mel spectrogram from content, pitch, speaker

    Args:
        content: Content features [B, C, T]
        pitch: Pitch features [B, C, T]
        speaker: Speaker embedding [B, gin_channels]
        mask: Padding mask [B, 1, T]
        temperature: Sampling temperature for expressiveness

    Returns:
        Mel spectrogram [B, n_mels, T]
    """
    # Combine content and pitch
    x = content + pitch
    x = self.pre_flow(x) * mask

    # Expand speaker embedding to sequence
    g = speaker.unsqueeze(-1).expand(-1, -1, x.size(2))

    # Sample from prior
    z = torch.randn_like(x) * temperature

    # Apply inverse flow (synthesis direction)
    for flow in reversed(self.flows):
        z = flow(z, x_cond=x, g=g, reverse=True)

    # Post-processing
    mel = self.post_flow(z) * mask

    return mel
```

**Normalizing Flows**:
- Residual Coupling Blocks for invertible transformations
- Speaker conditioning via FiLM (Feature-wise Linear Modulation)
- Temperature control for sampling diversity

**Training**:
```python
def compute_loss(
    self,
    content: torch.Tensor,
    pitch: torch.Tensor,
    speaker: torch.Tensor,
    target_mel: torch.Tensor,
    mask: torch.Tensor
) -> Dict[str, torch.Tensor]:
    """
    Compute flow matching loss

    Args:
        content: Content features
        pitch: Pitch features
        speaker: Speaker embedding
        target_mel: Target mel spectrogram
        mask: Padding mask

    Returns:
        Loss dictionary
    """
    # Forward flow (analysis direction)
    x = content + pitch
    x = self.pre_flow(x) * mask
    g = speaker.unsqueeze(-1).expand(-1, -1, x.size(2))

    # Transform target to latent
    z = target_mel
    for flow in self.flows:
        z, log_det = flow(z, x_cond=x, g=g, reverse=False)

    # Negative log-likelihood
    nll_loss = 0.5 * (z ** 2 + np.log(2 * np.pi)) - log_det
    nll_loss = torch.sum(nll_loss * mask) / torch.sum(mask)

    return {'nll_loss': nll_loss}
```

### 2.4 HiFiGAN Vocoder

**Location**: `src/auto_voice/models/hifigan.py`

**Purpose**: Convert mel spectrogram to waveform with high fidelity.

#### Architecture

```python
class HiFiGAN(nn.Module):
    def __init__(
        self,
        in_channels: int = 80,
        upsample_rates: List[int] = [8, 8, 2, 2],
        upsample_kernel_sizes: List[int] = [16, 16, 4, 4],
        resblock_kernel_sizes: List[int] = [3, 7, 11],
        resblock_dilation_sizes: List[List[int]] = [[1,3,5], [1,3,5], [1,3,5]]
    ):
        super().__init__()

        # Initial convolution
        self.conv_pre = nn.Conv1d(in_channels, 512, 7, padding=3)

        # Upsampling layers
        self.ups = nn.ModuleList()
        for i, (u, k) in enumerate(zip(upsample_rates, upsample_kernel_sizes)):
            self.ups.append(
                nn.ConvTranspose1d(
                    512 // (2 ** i),
                    512 // (2 ** (i + 1)),
                    k,
                    u,
                    padding=(k - u) // 2
                )
            )

        # Multi-receptive field fusion (MRF)
        self.resblocks = nn.ModuleList()
        for i in range(len(self.ups)):
            ch = 512 // (2 ** (i + 1))
            for j, (k, d) in enumerate(zip(resblock_kernel_sizes, resblock_dilation_sizes)):
                self.resblocks.append(ResBlock(ch, k, d))

        # Output convolution
        self.conv_post = nn.Conv1d(ch, 1, 7, padding=3)
```

**Forward Pass**:
```python
def forward(self, mel: torch.Tensor) -> torch.Tensor:
    """
    Generate waveform from mel spectrogram

    Args:
        mel: Mel spectrogram [B, n_mels, T]

    Returns:
        Waveform [B, 1, T * hop_length]
    """
    # Initial convolution
    x = self.conv_pre(mel)

    # Upsample and refine
    for i in range(len(self.ups)):
        x = F.leaky_relu(x, 0.1)
        x = self.ups[i](x)

        # Multi-receptive field fusion
        xs = None
        for j in range(len(self.resblocks) // len(self.ups)):
            if xs is None:
                xs = self.resblocks[i * len(self.resblocks) // len(self.ups) + j](x)
            else:
                xs += self.resblocks[i * len(self.resblocks) // len(self.ups) + j](x)
        x = xs / (len(self.resblocks) // len(self.ups))

    # Output activation
    x = F.leaky_relu(x, 0.1)
    x = self.conv_post(x)
    x = torch.tanh(x)

    return x
```

**Training Losses**:
- **Mel Reconstruction Loss**: L1 distance between target and generated mel
- **Feature Matching Loss**: Match intermediate discriminator features
- **Adversarial Loss**: Generator vs multi-scale discriminator

**Discriminator Architecture**:
```python
class MultiScaleDiscriminator(nn.Module):
    """Multi-scale discriminator for high-quality synthesis"""
    def __init__(self):
        super().__init__()
        self.discriminators = nn.ModuleList([
            ScaleDiscriminator(),  # Original scale
            ScaleDiscriminator(),  # 2x downsampled
            ScaleDiscriminator()   # 4x downsampled
        ])
        self.avgpools = nn.ModuleList([
            nn.Identity(),
            nn.AvgPool1d(4, 2, padding=2),
            nn.AvgPool1d(4, 2, padding=2)
        ])
```

## 3. Training Pipeline

### 3.1 Data Preparation

**Location**: `src/auto_voice/training/dataset.py`

**SingingVoiceDataset**:
```python
class SingingVoiceDataset(Dataset):
    def __init__(
        self,
        audio_dir: str,
        speaker_ids: List[str],
        segment_size: int = 8192,
        sample_rate: int = 44100,
        hop_length: int = 512,
        augment: bool = True
    ):
        self.audio_files = self._collect_files(audio_dir)
        self.segment_size = segment_size
        self.augment = augment

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        """
        Returns:
            audio: Waveform segment [segment_size]
            mel: Mel spectrogram [n_mels, T]
            f0: F0 contour [T]
            speaker_id: Speaker index
        """
        # Load audio
        audio = self._load_audio(self.audio_files[index])

        # Random segment
        if len(audio) > self.segment_size:
            start = random.randint(0, len(audio) - self.segment_size)
            audio = audio[start:start + self.segment_size]

        # Data augmentation
        if self.augment:
            audio = self._augment(audio)

        # Extract features
        mel = self._compute_mel(audio)
        f0 = self._extract_f0(audio)

        return {
            'audio': audio,
            'mel': mel,
            'f0': f0,
            'speaker_id': self.speaker_ids[index]
        }
```

**Data Augmentation**:
```python
def _augment(self, audio: np.ndarray) -> np.ndarray:
    """Apply data augmentation"""
    # Speed perturbation (0.9-1.1x)
    if random.random() < 0.5:
        speed_factor = random.uniform(0.9, 1.1)
        audio = librosa.effects.time_stretch(audio, rate=speed_factor)

    # Pitch shift (±2 semitones)
    if random.random() < 0.5:
        pitch_shift = random.randint(-2, 2)
        audio = librosa.effects.pitch_shift(
            audio, sr=self.sample_rate, n_steps=pitch_shift
        )

    # Volume perturbation (0.8-1.2x)
    if random.random() < 0.5:
        volume_factor = random.uniform(0.8, 1.2)
        audio = audio * volume_factor

    return audio
```

### 3.2 Training Loop

**Location**: `src/auto_voice/training/trainer.py`

**Trainer Architecture**:
```python
class SingingVoiceTrainer:
    def __init__(
        self,
        model: SingingVoiceConverter,
        vocoder: HiFiGAN,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: Dict[str, Any]
    ):
        self.model = model
        self.vocoder = vocoder
        self.train_loader = train_loader
        self.val_loader = val_loader

        # Optimizers
        self.optimizer_g = Adam(
            model.parameters(),
            lr=config['learning_rate'],
            betas=(0.8, 0.99)
        )
        self.optimizer_d = Adam(
            vocoder.discriminator.parameters(),
            lr=config['learning_rate'],
            betas=(0.8, 0.99)
        )

        # Schedulers
        self.scheduler_g = ExponentialLR(self.optimizer_g, gamma=0.999)
        self.scheduler_d = ExponentialLR(self.optimizer_d, gamma=0.999)
```

**Training Step**:
```python
def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
    """Single training step"""
    # Move to device
    audio = batch['audio'].to(self.device)
    mel = batch['mel'].to(self.device)
    f0 = batch['f0'].to(self.device)
    speaker_id = batch['speaker_id'].to(self.device)

    # Extract features
    content = self.model.content_encoder(mel)
    pitch = self.model.pitch_encoder(f0)
    speaker = self.model.speaker_encoder(mel)

    # Generate mel
    mel_pred = self.model.flow_decoder(content, pitch, speaker)

    # Generate audio
    audio_pred = self.vocoder(mel_pred)

    # Discriminator update
    self.optimizer_d.zero_grad()

    # Real samples
    y_d_r = self.vocoder.discriminator(audio)

    # Fake samples
    y_d_g = self.vocoder.discriminator(audio_pred.detach())

    # Adversarial loss
    loss_d = self._discriminator_loss(y_d_r, y_d_g)
    loss_d.backward()
    self.optimizer_d.step()

    # Generator update
    self.optimizer_g.zero_grad()

    # Adversarial loss
    y_d_g = self.vocoder.discriminator(audio_pred)
    loss_g_adv = self._generator_adversarial_loss(y_d_g)

    # Feature matching loss
    loss_fm = self._feature_matching_loss(y_d_r, y_d_g)

    # Mel reconstruction loss
    loss_mel = F.l1_loss(mel_pred, mel)

    # Flow matching loss
    loss_flow = self.model.flow_decoder.compute_loss(
        content, pitch, speaker, mel
    )['nll_loss']

    # Total generator loss
    loss_g = loss_g_adv + loss_fm + 45 * loss_mel + loss_flow
    loss_g.backward()

    # Gradient clipping
    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

    self.optimizer_g.step()

    return {
        'loss_d': loss_d.item(),
        'loss_g': loss_g.item(),
        'loss_mel': loss_mel.item(),
        'loss_flow': loss_flow.item()
    }
```

### 3.3 Hyperparameters

**Model Configuration**:
```yaml
model:
  content_encoder:
    hidden_channels: 192
    filter_channels: 768
    n_layers: 6
    n_heads: 2

  speaker_encoder:
    embedding_size: 256
    hidden_size: 256
    num_layers: 3

  pitch_encoder:
    n_vocab: 360
    out_channels: 192
    n_layers: 3

  flow_decoder:
    hidden_channels: 192
    n_flows: 4
    n_layers: 4

  vocoder:
    upsample_rates: [8, 8, 2, 2]
    resblock_kernel_sizes: [3, 7, 11]

training:
  batch_size: 16
  learning_rate: 0.0002
  betas: [0.8, 0.99]
  lr_decay: 0.999
  segment_size: 8192

  loss_weights:
    mel_weight: 45.0
    feature_matching_weight: 2.0

  epochs: 1000
  save_interval: 10
  validate_interval: 5
```

## 4. Inference Pipeline

### 4.1 SingingConversionPipeline

**Location**: `src/auto_voice/inference/singing_conversion_pipeline.py`

**Complete Pipeline**:
```python
class SingingConversionPipeline:
    def __init__(
        self,
        device: str = 'cuda',
        quality_preset: str = 'balanced'
    ):
        # Load models
        self.separator = VocalSeparator(device=device)
        self.pitch_extractor = SingingPitchExtractor(device=device)
        self.converter = SingingVoiceConverter.load_pretrained(device=device)
        self.vocoder = HiFiGAN.load_pretrained(device=device)
        self.speaker_encoder = SpeakerEncoder.load_pretrained(device=device)

    def convert_song(
        self,
        song_path: str,
        target_profile_id: str,
        vocal_volume: float = 1.0,
        instrumental_volume: float = 0.9,
        pitch_shift_semitones: int = 0,
        temperature: float = 1.0,
        return_stems: bool = False
    ) -> Dict[str, Any]:
        """
        Complete song conversion pipeline

        Pipeline stages:
        1. Separate vocals from instrumental (0-25%)
        2. Extract pitch contour from vocals (25-40%)
        3. Convert vocals to target voice (40-80%)
        4. Mix converted vocals with instrumental (80-100%)
        """
        # Stage 1: Separation (0-25%)
        stems = self.separator.separate(song_path)
        vocals = stems['vocals']
        instrumental = stems['instrumental']

        # Stage 2: Pitch extraction (25-40%)
        f0, confidence = self.pitch_extractor.extract_f0(vocals)

        # Apply pitch shift if requested
        if pitch_shift_semitones != 0:
            f0 = self._shift_pitch(f0, pitch_shift_semitones)

        # Stage 3: Voice conversion (40-80%)
        # Extract content from vocals
        mel_vocals = self._compute_mel(vocals)
        content = self.converter.content_encoder(mel_vocals)

        # Get target speaker embedding
        speaker_embedding = self._load_speaker_embedding(target_profile_id)

        # Encode pitch
        pitch_features = self.converter.pitch_encoder(f0)

        # Generate target mel
        mel_target = self.converter.flow_decoder(
            content,
            pitch_features,
            speaker_embedding,
            temperature=temperature
        )

        # Synthesize audio
        vocals_converted = self.vocoder(mel_target)

        # Stage 4: Mixing (80-100%)
        # Adjust volumes
        vocals_converted *= vocal_volume
        instrumental *= instrumental_volume

        # Mix
        mixed = vocals_converted + instrumental

        # Normalize
        mixed = self._normalize_audio(mixed)

        # Compute quality metrics
        metrics = self._compute_metrics(
            vocals, vocals_converted, f0, speaker_embedding
        )

        return {
            'output_path': output_path,
            'quality_metrics': metrics,
            'stems': {
                'vocals': vocals_converted if return_stems else None,
                'instrumental': instrumental if return_stems else None
            }
        }
```

### 4.2 TensorRT Optimization

**Location**: `src/auto_voice/inference/tensorrt_converter.py`

**Model Conversion**:
```python
class TensorRTConverter:
    """Convert PyTorch models to TensorRT for 2-3x speedup"""

    def convert_model(
        self,
        model: nn.Module,
        input_shapes: Dict[str, Tuple[int, ...]],
        fp16: bool = True,
        workspace_size: int = 1 << 30
    ) -> trt.ICudaEngine:
        """
        Convert PyTorch model to TensorRT engine

        Args:
            model: PyTorch model
            input_shapes: Dictionary of input names to shapes
            fp16: Use FP16 precision for 2x speedup
            workspace_size: Maximum workspace size (1GB default)

        Returns:
            TensorRT engine
        """
        # Export to ONNX
        onnx_path = self._export_onnx(model, input_shapes)

        # Build TensorRT engine
        logger = trt.Logger(trt.Logger.WARNING)
        builder = trt.Builder(logger)
        network = builder.create_network(
            1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
        )
        parser = trt.OnnxParser(network, logger)

        # Parse ONNX
        with open(onnx_path, 'rb') as f:
            parser.parse(f.read())

        # Build config
        config = builder.create_builder_config()
        config.max_workspace_size = workspace_size

        if fp16:
            config.set_flag(trt.BuilderFlag.FP16)

        # Build engine
        engine = builder.build_engine(network, config)

        return engine
```

**Runtime Inference**:
```python
class TensorRTInference:
    """TensorRT inference engine wrapper"""

    def __init__(self, engine_path: str):
        # Load engine
        with open(engine_path, 'rb') as f:
            self.engine = trt.Runtime(trt.Logger()).deserialize_cuda_engine(f.read())

        self.context = self.engine.create_execution_context()

    def infer(self, inputs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Run inference with TensorRT engine

        Args:
            inputs: Dictionary of input names to numpy arrays

        Returns:
            Dictionary of output names to numpy arrays
        """
        # Allocate buffers
        bindings = []
        outputs = {}

        for i in range(self.engine.num_bindings):
            name = self.engine.get_binding_name(i)
            dtype = trt.nptype(self.engine.get_binding_dtype(i))
            shape = self.engine.get_binding_shape(i)

            if self.engine.binding_is_input(i):
                buffer = cuda.mem_alloc(inputs[name].nbytes)
                cuda.memcpy_htod(buffer, inputs[name])
            else:
                buffer = cuda.mem_alloc(
                    trt.volume(shape) * np.dtype(dtype).itemsize
                )
                outputs[name] = np.empty(shape, dtype=dtype)

            bindings.append(int(buffer))

        # Run inference
        self.context.execute_v2(bindings)

        # Copy outputs
        for name in outputs:
            cuda.memcpy_dtoh(outputs[name], bindings[name])

        return outputs
```

## 5. Quality Metrics

### 5.1 Pitch Accuracy

**RMSE (Root Mean Square Error)**:
```python
def compute_pitch_rmse_hz(
    reference_f0: np.ndarray,
    converted_f0: np.ndarray
) -> float:
    """
    Compute pitch RMSE in Hz domain

    Target: < 10 Hz (imperceptible to most listeners)
    """
    # Filter unvoiced frames
    voiced_mask = (reference_f0 > 0) & (converted_f0 > 0)

    # Compute RMSE
    rmse = np.sqrt(np.mean(
        (reference_f0[voiced_mask] - converted_f0[voiced_mask]) ** 2
    ))

    return rmse
```

**F0 Correlation**:
```python
def compute_f0_correlation(
    reference_f0: np.ndarray,
    converted_f0: np.ndarray
) -> float:
    """
    Compute Pearson correlation of F0 contours

    Target: > 0.9
    """
    from scipy.stats import pearsonr

    voiced_mask = (reference_f0 > 0) & (converted_f0 > 0)
    correlation, _ = pearsonr(
        reference_f0[voiced_mask],
        converted_f0[voiced_mask]
    )

    return correlation
```

### 5.2 Speaker Similarity

**Cosine Similarity**:
```python
def compute_speaker_similarity(
    target_embedding: np.ndarray,
    converted_embedding: np.ndarray
) -> float:
    """
    Compute cosine similarity between speaker embeddings

    Target: > 0.85 (85% match)
    """
    from sklearn.metrics.pairwise import cosine_similarity

    similarity = cosine_similarity(
        target_embedding.reshape(1, -1),
        converted_embedding.reshape(1, -1)
    )[0, 0]

    return similarity
```

### 5.3 Audio Quality

**Mel Cepstral Distortion (MCD)**:
```python
def compute_mcd(
    reference_mel: np.ndarray,
    converted_mel: np.ndarray
) -> float:
    """
    Compute Mel Cepstral Distortion

    Target: < 6.0 dB
    """
    # Convert to cepstral domain
    from scipy.fft import dct
    reference_cep = dct(reference_mel, axis=0, norm='ortho')
    converted_cep = dct(converted_mel, axis=0, norm='ortho')

    # Compute MCD (excluding 0th coefficient)
    mcd = np.sqrt(np.sum((reference_cep[1:] - converted_cep[1:]) ** 2, axis=0))
    mcd = (10 / np.log(10)) * np.mean(mcd)

    return mcd
```

**STOI (Short-Time Objective Intelligibility)**:
```python
def compute_stoi(
    reference_audio: np.ndarray,
    converted_audio: np.ndarray,
    sample_rate: int = 44100
) -> float:
    """
    Compute STOI for intelligibility assessment

    Target: > 0.9
    """
    from pystoi import stoi

    score = stoi(
        reference_audio,
        converted_audio,
        sample_rate,
        extended=False
    )

    return score
```

## 6. Performance Optimizations

### 6.1 GPU Acceleration

**CUDA Kernels** (`src/cuda_kernels/`):
- **Pitch Detection**: Custom CUDA implementation of CREPE for 5x speedup
- **Vibrato Analysis**: Fast Fourier Transform on GPU
- **Mel Computation**: GPU-accelerated mel spectrogram
- **Audio Mixing**: Parallel waveform operations

**Mixed Precision Training**:
```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

with autocast():
    # Forward pass in FP16
    loss = compute_loss(model, batch)

# Backward in FP16, update in FP32
scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

### 6.2 Caching Strategies

**Vocal Separation Cache**:
```python
def get_cached_separation(
    audio_path: str,
    cache_dir: str
) -> Optional[Dict[str, str]]:
    """Check for cached separation results"""
    cache_key = hashlib.sha256(
        open(audio_path, 'rb').read()
    ).hexdigest()

    vocals_path = os.path.join(cache_dir, f"{cache_key}_vocals.wav")
    instrumental_path = os.path.join(cache_dir, f"{cache_key}_instrumental.wav")

    if os.path.exists(vocals_path) and os.path.exists(instrumental_path):
        return {
            'vocals': vocals_path,
            'instrumental': instrumental_path
        }

    return None
```

**Conversion Cache**:
- Cache speaker embeddings by profile ID
- Cache pitch contours by audio hash
- Cache mel spectrograms during batch processing

### 6.3 Batch Processing

**Parallel Song Conversion**:
```python
def batch_convert_songs(
    songs: List[str],
    target_profile_id: str,
    num_workers: int = 4
) -> List[Dict[str, Any]]:
    """Convert multiple songs in parallel"""
    from concurrent.futures import ProcessPoolExecutor

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = [
            executor.submit(convert_song, song, target_profile_id)
            for song in songs
        ]

        results = [future.result() for future in futures]

    return results
```

## 7. Model Checkpoints

### 7.1 Pre-trained Models

**Available Checkpoints**:
- `models/content_encoder.pt`: Content encoder weights
- `models/speaker_encoder.pt`: Speaker encoder (Resemblyzer)
- `models/pitch_encoder.pt`: Pitch encoder weights
- `models/flow_decoder.pt`: Flow decoder weights
- `models/hifigan_vocoder.pt`: HiFiGAN vocoder

**Loading Pre-trained Models**:
```python
from auto_voice.models import SingingVoiceConverter

model = SingingVoiceConverter.load_pretrained(
    checkpoint_path='models/singing_voice_converter.pt',
    device='cuda'
)
```

### 7.2 Model Export

**ONNX Export**:
```python
def export_onnx(
    model: nn.Module,
    output_path: str,
    input_shapes: Dict[str, Tuple[int, ...]]
):
    """Export model to ONNX format"""
    # Create dummy inputs
    dummy_inputs = {
        name: torch.randn(*shape).to(model.device)
        for name, shape in input_shapes.items()
    }

    # Export
    torch.onnx.export(
        model,
        tuple(dummy_inputs.values()),
        output_path,
        input_names=list(dummy_inputs.keys()),
        output_names=['output'],
        dynamic_axes={
            name: {0: 'batch_size', 2: 'sequence_length'}
            for name in dummy_inputs.keys()
        },
        opset_version=13
    )
```

## 8. References

**So-VITS-SVC**:
- SoftVC VITS Singing Voice Conversion: https://github.com/svc-develop-team/so-vits-svc

**HiFiGAN**:
- HiFi-GAN: Generative Adversarial Networks for Efficient and High Fidelity Speech Synthesis
- Kong et al., NeurIPS 2020

**Demucs**:
- Hybrid Spectrogram and Waveform Source Separation
- Défossez et al., 2021

**Torchcrepe**:
- CREPE: A Convolutional Representation for Pitch Estimation
- Kim et al., ICASSP 2018

**Resemblyzer**:
- Transfer Learning from Speaker Verification to Multispeaker Text-To-Speech Synthesis
- Jia et al., NeurIPS 2018
