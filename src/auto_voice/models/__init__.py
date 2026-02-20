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
]
