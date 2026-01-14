# AutoVoice Completion - Jetson Thor Ralph Loop Orchestration

## MISSION

Complete AutoVoice to 100% production-ready on NVIDIA Jetson AGX Thor (sm_110) using iterative Ralph Loop orchestration with Beads task management. Run continuously until ALL features work and ALL tests pass.

---

## MORPHOLOGICAL SELF-HEALING PROTOCOL

When Ralph Loop encounters any failure (tool, dependency, test, MCP server):

```
┌─────────────────────────────────────────────────────────────┐
│              MORPHOLOGICAL SELF-HEALING                     │
│                                                             │
│  1. DETECT → Tool/dependency/test failure                   │
│  2. DIAGNOSE → Research root cause using scholarly tools    │
│  3. CREATE BEAD → bd create --title="Fix: [issue]"          │
│  4. IMPLEMENT → Apply fix using available tools             │
│  5. TEST → Verify fix works                                 │
│  6. CLOSE BEAD → bd close [id]                              │
│  7. CONTINUE → Resume original task                         │
└─────────────────────────────────────────────────────────────┘
```

### Self-Healing Examples:
- **MCP tool fails**: Create bead, research fix, repair MCP server, continue
- **Dependency missing**: Create bead, check facefusion env, copy or build from source
- **Test fails**: Create bead, diagnose, fix code, verify, close bead

---

## RESEARCH-FIRST PROTOCOL

For cutting-edge implementations, ALWAYS research before coding:

### Research Tools Available:
```
mcp__perplexity__perplexity_research - Deep research with citations
mcp__arxiv-advanced__search_papers - Academic papers
mcp__tavily__tavily-search - Web search
mcp__research-semantic-scholar__search_semantic_scholar - Scholarly papers
mcp__paper-search__search_papers - Multi-source paper search
```

### Research Phase Template:
```
1. Query: [specific technical question]
2. Sources: arXiv, Semantic Scholar, Perplexity
3. Findings: [key insights]
4. Decision: [implementation choice with rationale]
5. Store in Cipher: mcp__cipher__ask_cipher "Store: [findings]"
```

---

## MEMORY LAYERS

### Cipher - Cross-Session Memory
```
# Store research findings
mcp__cipher__ask_cipher "Store: Thor sm_110 requires CUDA 13.0 with -std=c++17 flag"

# Query previous knowledge
mcp__cipher__ask_cipher "Query: What were the Thor kernel compilation fixes?"
```

### Serena - Code Intelligence
```
mcp__serena__activate_project project="/home/kp/repos/autovoice"
mcp__serena__find_symbol name_path_pattern="SingingConversionPipeline"
mcp__serena__get_symbols_overview relative_path="src/auto_voice/inference"
mcp__serena__replace_symbol_body name_path="ClassName/method_name" body="..."
```

### Beads - Task Management (MCP + CLI)
```
# MCP tools
mcp__beads__set_context workspace_root="/home/kp/repos/autovoice"
mcp__beads__create title="Fix: issue" issue_type="bug" priority=1
mcp__beads__list status="open"
mcp__beads__close issue_id="AV-xxx"

# CLI fallback (if MCP fails)
bd ready           # Get available tasks
bd show [id]       # Task details
bd update [id] --status=in_progress
bd close [id]      # Complete task
bd sync            # Push changes
```

---

## PLATFORM SPECIFICATIONS

```yaml
Hardware:
  GPU: NVIDIA Thor (Compute Capability 11.0, sm_110)
  Architecture: aarch64
  JetPack: R38.4.0
  Driver: 580.00
  CUDA: 13.0

Environment:
  Conda: autovoice-thor
  Python: 3.12.12
  PyTorch: 2.11.0.dev20260113+cu130
  Working Directory: /home/kp/repos/autovoice

Services:
  MySQL Password: teamrsi123teamrsi123teamrsi123
  Available Ports: 5000, 8000, 3000, 8080
  Reserved: Check with lsof before binding
```

---

## CRITICAL RULES

### DO:
1. **GPU-ONLY Dependencies**: All CUDA/TensorRT packages MUST use GPU implementations, NEVER CPU fallbacks
2. **Copy Dependencies**: When missing packages, copy from facefusion or other working conda envs first
3. **Compile from Source**: If not in pip registries, build from source for aarch64
4. **Create Separate Envs**: If conflicts arise, create new conda environments to isolate
5. **Check Ports First**: Run `lsof -i :PORT` before binding any service
6. **Test Every Feature**: Iterate through EVERY module, function, and API endpoint
7. **Create Beads Dynamically**: When problems found, immediately create beads issues
8. **Run Forever**: Loop until 100% complete - no early exit

### DON'T:
1. **Never Modify Foreign Envs**: Do NOT change dependency trees in conda environments that aren't native to this project (facefusion, base, etc.)
2. **Never Use CPU Fallbacks**: If GPU version unavailable, compile from source - don't fall back to CPU
3. **Never Destroy Services**: Check what's running before killing processes
4. **Never Skip Tests**: Every test must pass or have documented skip reason
5. **Never Hardcode Ports**: Always check availability first

---

## DEPENDENCY RESOLUTION STRATEGY

### Priority Order:
```
1. pip install from PyPI (with --index-url for CUDA wheels)
2. Copy from facefusion conda env: /home/kp/anaconda3/envs/facefusion/
3. Copy from base conda env: /home/kp/anaconda3/
4. Build from source with CUDA support
5. Create isolated conda env for conflicting packages
```

### Copying Dependencies (Safe Method):
```bash
# Find package location in source env
FACEFUSION_SITE="/home/kp/anaconda3/envs/facefusion/lib/python3.12/site-packages"

# Copy without modifying source
cp -r "$FACEFUSION_SITE/package_name" \
      "/home/kp/anaconda3/envs/autovoice-thor/lib/python3.12/site-packages/"

# NEVER run pip install/uninstall in foreign environments
```

### Building from Source (aarch64):
```bash
# Clone and build for Jetson
git clone https://github.com/org/package.git
cd package
export CUDA_HOME=/usr/local/cuda-13.0
export TORCH_CUDA_ARCH_LIST="11.0"
pip install -e . --no-build-isolation
```

---

## RALPH LOOP PROTOCOL

### Research-First Infinite Execution Loop:
```
┌─────────────────────────────────────────────────────────────┐
│          RESEARCH-FIRST SELF-HEALING RALPH LOOP             │
│                                                             │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐              │
│  │  SCAN    │ →  │ RESEARCH │ →  │   TEST   │              │
│  │ bd ready │    │ if needed│    │  pytest  │              │
│  └────┬─────┘    └────┬─────┘    └────┬─────┘              │
│       │               │               │                     │
│       │               ▼               ▼                     │
│       │         ┌──────────┐    ┌──────────┐              │
│       │         │  STORE   │    │   FIX    │              │
│       │         │  Cipher  │    │ if fail  │              │
│       │         └──────────┘    └────┬─────┘              │
│       │                              │                     │
│       ▼               ▼              ▼                     │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐              │
│  │  CREATE  │ ←  │  VERIFY  │ ←  │  COMMIT  │              │
│  │  BEADS   │    │  WORKS   │    │  CHANGES │              │
│  └──────────┘    └──────────┘    └──────────┘              │
│       │                                                     │
│       └──────────── REPEAT FOREVER ─────────────────────→  │
└─────────────────────────────────────────────────────────────┘
```

### Loop Stages:

1. **SCAN**: Get next task from beads (`bd ready`)
2. **RESEARCH**: If cutting-edge issue, use research tools:
   - `mcp__perplexity__perplexity_research` for deep research
   - `mcp__arxiv-advanced__search_papers` for academic papers
   - Store findings: `mcp__cipher__ask_cipher "Store: [findings]"`
3. **TEST**: Execute tests for current feature
4. **FIX**: If failure, apply morphological self-healing:
   - Create bead: `bd create --title="Fix: [issue]"`
   - Research solution
   - Implement fix using Serena
   - Verify fix
   - Close bead: `bd close [id]`
5. **COMMIT**: Save working state to git + `bd sync`
6. **VERIFY**: Confirm fix actually works
7. **CREATE BEADS**: Track any new issues discovered
8. **REPEAT**: Continue until 100% complete

---

## FEATURE TESTING CHECKLIST

### Phase 1: Environment & Dependencies
```yaml
BEAD-ENV-001:
  name: "Verify PyTorch CUDA on Thor"
  test: |
    python -c "import torch; assert torch.cuda.is_available(); \
               x = torch.randn(1000,1000).cuda(); \
               print(f'GPU: {torch.cuda.get_device_name()}')"
  acceptance: "CUDA tensor ops work on sm_110"

BEAD-ENV-002:
  name: "Install Core Dependencies"
  test: |
    pip install numpy librosa soundfile scipy matplotlib flask flask-socketio
    python -c "import librosa; import flask; print('Core deps OK')"
  acceptance: "All core packages import"

BEAD-ENV-003:
  name: "Install ML Dependencies"
  test: |
    pip install transformers webrtcvad noisereduce resemblyzer
    # If fairseq fails, build from source:
    # git clone https://github.com/facebookresearch/fairseq && cd fairseq
    # pip install --no-deps -e .
  acceptance: "ML packages available"

BEAD-ENV-004:
  name: "Install Audio Processing"
  test: |
    pip install torchcrepe praat-parselmouth pyrubberband
    python -c "import torchcrepe; print('Pitch detection ready')"
  acceptance: "Audio processing libs work"

BEAD-ENV-005:
  name: "Install Demucs (Source Separation)"
  test: |
    pip install --no-deps demucs
    python -c "import demucs; print('Demucs ready')"
  acceptance: "Source separation available"
```

### Phase 2: CUDA Kernels
```yaml
BEAD-CUDA-001:
  name: "Build CUDA Extensions"
  test: |
    cd /home/kp/repos/autovoice
    export TORCH_CUDA_ARCH_LIST="11.0"
    pip install -e .
    python -c "import cuda_kernels; print(dir(cuda_kernels))"
  acceptance: "cuda_kernels module imports"

BEAD-CUDA-002:
  name: "Test Pitch Detection Kernel"
  test: |
    python -c "
    import torch
    import cuda_kernels
    audio = torch.randn(16000).cuda()
    result = cuda_kernels.detect_pitch(audio, 16000)
    print(f'Pitch shape: {result.shape}')
    "
  acceptance: "Pitch kernel executes on GPU"

BEAD-CUDA-003:
  name: "Test FFT Kernels"
  test: |
    pytest tests/gpu/test_fft_kernels.py -v
  acceptance: "FFT tests pass"

BEAD-CUDA-004:
  name: "Test Training Kernels"
  test: |
    pytest tests/gpu/test_training_kernels.py -v
  acceptance: "Training kernel tests pass"
```

### Phase 3: Audio Processing
```yaml
BEAD-AUDIO-001:
  name: "Audio Loader Tests"
  test: pytest tests/test_audio_processor.py -v -k "load"
  acceptance: "Audio loading works"

BEAD-AUDIO-002:
  name: "Mel Spectrogram Tests"
  test: pytest tests/test_audio_processor.py -v -k "mel"
  acceptance: "Mel extraction works"

BEAD-AUDIO-003:
  name: "Pitch Extraction Tests"
  test: pytest tests/test_audio_processor.py -v -k "pitch"
  acceptance: "Pitch extraction works"

BEAD-AUDIO-004:
  name: "Source Separation Tests"
  test: |
    python -c "
    from auto_voice.audio.source_separator import SourceSeparator
    sep = SourceSeparator()
    print('Source separator initialized')
    "
  acceptance: "Demucs separation ready"
```

### Phase 4: Models
```yaml
BEAD-MODEL-001:
  name: "Encoder Tests"
  test: pytest tests/models/test_encoder.py -v
  acceptance: "Encoder model works"

BEAD-MODEL-002:
  name: "Vocoder Tests"
  test: pytest tests/models/test_vocoder.py -v
  acceptance: "HiFiGAN vocoder works"

BEAD-MODEL-003:
  name: "Voice Transformer Tests"
  test: pytest tests/test_models.py -v -k "transformer"
  acceptance: "Voice transformer works"

BEAD-MODEL-004:
  name: "Speaker Encoder Tests"
  test: |
    python -c "
    from auto_voice.inference.voice_cloner import VoiceCloner
    cloner = VoiceCloner()
    print('Voice cloner ready')
    "
  acceptance: "Speaker encoder initialized"
```

### Phase 5: Inference Pipeline
```yaml
BEAD-INF-001:
  name: "Voice Cloning Pipeline"
  test: pytest tests/inference/test_voice_cloner.py -v
  acceptance: "Voice cloning works"

BEAD-INF-002:
  name: "Singing Conversion Pipeline"
  test: pytest tests/test_singing_conversion.py -v
  acceptance: "Singing conversion works"

BEAD-INF-003:
  name: "Real-time Conversion"
  test: pytest tests/inference/test_realtime.py -v
  acceptance: "Real-time pipeline works"

BEAD-INF-004:
  name: "End-to-End Conversion"
  test: |
    python -c "
    from auto_voice.inference.singing_conversion_pipeline import SingingConversionPipeline
    pipeline = SingingConversionPipeline()
    print('Full pipeline ready')
    "
  acceptance: "E2E pipeline initializes"
```

### Phase 6: Web API
```yaml
BEAD-WEB-001:
  name: "Flask App Creation"
  test: |
    python -c "
    from auto_voice.web.app import create_app
    app = create_app()
    print('Flask app created')
    "
  acceptance: "App factory works"

BEAD-WEB-002:
  name: "Health Endpoint"
  test: |
    # Start server on available port
    PORT=$(python -c "import socket; s=socket.socket(); s.bind(('',0)); print(s.getsockname()[1]); s.close()")
    timeout 10 python main.py --port $PORT &
    sleep 5
    curl -f http://localhost:$PORT/health
    pkill -f "main.py --port $PORT"
  acceptance: "Health returns 200"

BEAD-WEB-003:
  name: "API Endpoints Test"
  test: pytest tests/test_api_e2e_validation.py -v
  acceptance: "API endpoints work"

BEAD-WEB-004:
  name: "WebSocket Connection"
  test: |
    python scripts/test_websocket_connection.py
  acceptance: "WebSocket connects"
```

### Phase 7: Frontend
```yaml
BEAD-FE-001:
  name: "Frontend Build"
  test: |
    cd frontend
    npm install
    npm run build
  acceptance: "Frontend builds without errors"

BEAD-FE-002:
  name: "TypeScript Check"
  test: |
    cd frontend
    npm run typecheck || npx tsc --noEmit
  acceptance: "No TypeScript errors"

BEAD-FE-003:
  name: "Frontend Dev Server"
  test: |
    cd frontend
    PORT=3001 timeout 30 npm run dev &
    sleep 10
    curl -f http://localhost:3001
    pkill -f "vite"
  acceptance: "Dev server starts"
```

### Phase 8: Integration Tests
```yaml
BEAD-INT-001:
  name: "Full Test Suite"
  test: pytest tests/ -v --ignore=tests/gpu
  acceptance: ">90% tests pass"

BEAD-INT-002:
  name: "GPU Test Suite"
  test: pytest tests/gpu/ -v
  acceptance: "GPU tests pass"

BEAD-INT-003:
  name: "Integration Tests"
  test: pytest tests/integration/ -v
  acceptance: "Integration tests pass"

BEAD-INT-004:
  name: "Performance Tests"
  test: pytest tests/test_performance.py -v
  acceptance: "Performance acceptable"
```

### Phase 9: Docker & Deployment
```yaml
BEAD-DEPLOY-001:
  name: "Docker Build"
  test: |
    docker build -t autovoice:thor -f Dockerfile.jetson .
  acceptance: "Docker image builds"

BEAD-DEPLOY-002:
  name: "Docker Run with GPU"
  test: |
    docker run --runtime nvidia --gpus all -d \
      -p 5050:5000 autovoice:thor
    sleep 10
    curl -f http://localhost:5050/health
    docker stop $(docker ps -q --filter ancestor=autovoice:thor)
  acceptance: "Container runs with GPU"
```

---

## DYNAMIC BEAD CREATION

When a test fails, immediately create a bead:

```bash
# Use beads MCP to track issues
mcp__beads__set_context workspace_root="/home/kp/repos/autovoice"

# Create issue for failure
mcp__beads__create \
  title="Fix: [COMPONENT] - [ERROR_SUMMARY]" \
  issue_type="bug" \
  priority=1 \
  description="Test failed: [TEST_NAME]
Error: [ERROR_MESSAGE]
Stack trace: [TRACE]

Attempted fixes:
- [LIST]

Next steps:
- [ACTIONS]"
```

---

## PORT MANAGEMENT

### Before Starting Any Service:
```bash
# Check if port is in use
check_port() {
    local port=$1
    if lsof -i :$port > /dev/null 2>&1; then
        echo "Port $port is IN USE"
        lsof -i :$port
        return 1
    else
        echo "Port $port is AVAILABLE"
        return 0
    fi
}

# Find available port
find_port() {
    local base=${1:-5000}
    for port in $(seq $base $((base+100))); do
        if ! lsof -i :$port > /dev/null 2>&1; then
            echo $port
            return 0
        fi
    done
    return 1
}
```

### Reserved Ports (DO NOT USE):
- 22: SSH
- 80/443: HTTP/HTTPS
- 3306: MySQL (password: teamrsi123teamrsi123teamrsi123)
- Check current usage: `sudo lsof -i -P -n | grep LISTEN`

---

## COMPLETION CRITERIA

### 100% Complete When:
1. [ ] All conda dependencies installed (GPU versions only)
2. [ ] CUDA kernels compile and run on sm_110
3. [ ] All audio processing tests pass
4. [ ] All model tests pass
5. [ ] Inference pipeline works end-to-end
6. [ ] Web API responds correctly
7. [ ] WebSocket real-time updates work
8. [ ] Frontend builds and connects to backend
9. [ ] Docker container runs with GPU
10. [ ] Full test suite passes (>95%)
11. [ ] Demo song conversion successful
12. [ ] Setup scripts created for Jetson Thor

### Exit Condition:
**DO NOT EXIT until ALL criteria are met.** If blocked, create beads and continue to next feature. Return to blocked items after resolving dependencies.

---

## SETUP SCRIPT GENERATION

When complete, generate these scripts:

### scripts/setup_jetson_thor.sh
```bash
#!/bin/bash
# AutoVoice Setup for NVIDIA Jetson AGX Thor
# Generated by Ralph Loop orchestration

set -e

echo "=== AutoVoice Jetson Thor Setup ==="

# Create conda environment
conda create -n autovoice-thor python=3.12 -y
conda activate autovoice-thor

# Install PyTorch for CUDA 13.0
pip install --pre torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/nightly/cu130

# Install dependencies
pip install -r requirements-jetson.txt

# Build CUDA extensions for sm_110
export TORCH_CUDA_ARCH_LIST="11.0"
pip install -e .

# Verify installation
python -c "import torch; assert torch.cuda.is_available()"
python -c "import cuda_kernels; print('CUDA kernels ready')"

echo "=== Setup Complete ==="
```

### requirements-jetson.txt
```
# Generated requirements for Jetson Thor (aarch64, CUDA 13.0)
# Core dependencies validated on R38.4.0
numpy>=1.26,<2.0
librosa>=0.10,<0.11
soundfile>=0.12,<0.13
scipy>=1.12,<1.14
flask>=2.3,<3.0
flask-socketio>=5.3,<6.0
flask-cors>=4.0,<5.0
# ... (complete list generated during run)
```

---

## BROWSER AUTOMATION TESTING

Use Playwright MCP for end-to-end browser testing:

```
# Navigate to application
mcp__playwright__playwright_navigate url="http://localhost:5000"

# Take screenshot
mcp__playwright__playwright_screenshot name="home_page"

# Test upload flow
mcp__playwright__playwright_click selector="#upload-button"
mcp__playwright__playwright_fill selector="#file-input" value="test.mp3"

# Verify conversion
mcp__playwright__playwright_get_visible_text
mcp__playwright__playwright_screenshot name="conversion_result"
```

### Browser Test Checklist:
1. [ ] Home page loads correctly
2. [ ] Upload form works
3. [ ] Voice profile selection works
4. [ ] Conversion progress displays
5. [ ] Download link appears after conversion
6. [ ] Error messages display correctly

---

## CREDENTIALS & RESOURCES

```yaml
NGC API Key: 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
MySQL Password: teamrsi123teamrsi123teamrsi123
Working Directory: /home/kp/repos/autovoice
Conda Environment: autovoice-thor

Ports In Use (DO NOT USE):
  - 3333, 5911, 10000, 10003, 10004, 10005
  - 12200, 12960, 24282-24286, 37373

Available Ports:
  - 5000 (Flask API)
  - 8080 (Frontend)
  - 8000 (Alternative API)
```

---

## START EXECUTION

Begin Ralph Loop now. Execute each bead in order. Create new beads for any failures. Do not stop until 100% complete.

**Phase 0**: Deep Research (research beads first)
- AV-054: Jetson Thor sm_110 CUDA 13.0 compatibility research
- AV-1ty: So-VITS-SVC 5.0 architecture analysis
- AV-8me: aarch64 dependency availability research

**Phase 1**: Environment & Dependencies
- Verify autovoice-thor environment
- Run BEAD-ENV-001

**Then continue through all phases until 100% complete.**
