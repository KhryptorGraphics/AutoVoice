"""Neural network model architectures."""
from .encoder import ContentEncoder, PitchEncoder, HuBERTSoft
from .vocoder import HiFiGANVocoder, HiFiGANGenerator
from .so_vits_svc import SoVitsSvc
from .consistency import (
    DiffusionDecoder, ConsistencyStudent, CTLoss_D, EDMLoss,
    KarrasNoiseSchedule, ResidualBlock, DiffusionStepEmbedding,
)
from .svc_decoder import CoMoSVCDecoder, BiDilConv, FiLMConditioning
from .smoothsinger_decoder import (
    SmoothSingerDecoder, MultiResolutionBlock, DualBranchFusion,
)
from .nsf_module import NSFHarmonicEnhancer
from .pupu_vocoder import PupuVocoderEnhancer
from .ecapa2_encoder import ECAPA2SpeakerEncoder, ECAPA2EmbeddingResult

__all__ = [
    'ContentEncoder',
    'PitchEncoder',
    'HuBERTSoft',
    'HiFiGANVocoder',
    'HiFiGANGenerator',
    'SoVitsSvc',
    'DiffusionDecoder',
    'ConsistencyStudent',
    'CTLoss_D',
    'EDMLoss',
    'KarrasNoiseSchedule',
    'ResidualBlock',
    'DiffusionStepEmbedding',
    'CoMoSVCDecoder',
    'BiDilConv',
    'FiLMConditioning',
    'SmoothSingerDecoder',
    'MultiResolutionBlock',
    'DualBranchFusion',
    'NSFHarmonicEnhancer',
    'PupuVocoderEnhancer',
    'ECAPA2SpeakerEncoder',
    'ECAPA2EmbeddingResult',
]
