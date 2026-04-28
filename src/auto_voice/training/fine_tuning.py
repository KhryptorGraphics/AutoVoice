"""Fine-tuning pipeline with LoRA adapters and EWC regularization.

Implements SOTA techniques from research:
- LoRA adapters for parameter-efficient fine-tuning
- EWC (Elastic Weight Consolidation) for catastrophic forgetting prevention
- Layer freezing strategies

Task 4.4: Implement fine-tuning pipeline (freeze layers, train adapter/LoRA, full fine-tune options)
"""

import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from .artifacts import (
    build_lora_checkpoint_payload,
    extract_lora_metadata,
    extract_lora_state_dict,
)

logger = logging.getLogger(__name__)


class LoRAAdapter(nn.Module):
    """Low-Rank Adaptation (LoRA) adapter for efficient fine-tuning."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 8,
        alpha: int = 16,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        self.lora_A = nn.Parameter(torch.empty(rank, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))

        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.dropout(x)
        return (x @ self.lora_A.T @ self.lora_B.T) * self.scaling


class LoRALinear(nn.Module):
    """Linear layer wrapped with LoRA adapter for parameter-efficient fine-tuning."""

    def __init__(
        self,
        original_linear: nn.Linear,
        rank: int = 8,
        alpha: int = 16,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.original = original_linear
        self.adapter = LoRAAdapter(
            in_features=original_linear.in_features,
            out_features=original_linear.out_features,
            rank=rank,
            alpha=alpha,
            dropout=dropout,
        )
        # Move adapter to same device as original layer
        device = original_linear.weight.device
        self.adapter = self.adapter.to(device)

        # Freeze original weights
        self.original.weight.requires_grad = False
        if self.original.bias is not None:
            self.original.bias.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Original frozen output + trainable LoRA contribution
        return self.original(x) + self.adapter(x)


def inject_lora_adapters(
    model: nn.Module,
    target_modules: List[str],
    rank: int = 8,
    alpha: int = 16,
    dropout: float = 0.0,
) -> nn.Module:
    """Inject LoRA adapters into target modules of a model.

    Replaces target Linear modules with LoRALinear wrappers that combine
    frozen original weights with trainable low-rank adapters.
    """
    model.lora_adapters = nn.ModuleDict()

    # Collect modules to replace (can't modify during iteration)
    replacements = []
    for name, module in model.named_modules():
        module_name = name.split(".")[-1]
        if module_name in target_modules and isinstance(module, nn.Linear):
            replacements.append((name, module))

    # Replace Linear modules with LoRALinear wrappers
    for name, original_linear in replacements:
        lora_linear = LoRALinear(
            original_linear=original_linear,
            rank=rank,
            alpha=alpha,
            dropout=dropout,
        )
        # Navigate to parent and replace the module
        parts = name.split(".")
        if len(parts) == 1:
            setattr(model, name, lora_linear)
        else:
            parent = model
            for part in parts[:-1]:
                parent = getattr(parent, part)
            setattr(parent, parts[-1], lora_linear)

        # Store reference for easy access
        safe_name = name.replace(".", "_")
        model.lora_adapters[safe_name] = lora_linear.adapter
        logger.debug(f"Injected LoRA adapter into {name}")

    model._lora_config = {
        "rank": rank,
        "alpha": alpha,
        "target_modules": target_modules,
        "dropout": dropout,
    }
    return model


def freeze_base_model(model: nn.Module) -> None:
    """Freeze all parameters in a model."""
    for param in model.parameters():
        param.requires_grad = False


def freeze_except(model: nn.Module, unfrozen_modules: List[str]) -> None:
    """Freeze all parameters except specified modules."""
    freeze_base_model(model)
    for name, module in model.named_modules():
        module_name = name.split(".")[-1]
        if module_name in unfrozen_modules:
            for param in module.parameters():
                param.requires_grad = True


def unfreeze_model(model: nn.Module) -> None:
    """Unfreeze all parameters in a model."""
    for param in model.parameters():
        param.requires_grad = True


def compute_fisher_information(
    model: nn.Module,
    dataloader: List[Tuple[torch.Tensor, torch.Tensor]],
    device: torch.device = None,
) -> Dict[str, torch.Tensor]:
    """Compute Fisher information matrix for EWC."""
    if device is None:
        device = next(model.parameters()).device

    model.eval()
    fisher_dict = {}
    for name, param in model.named_parameters():
        fisher_dict[name] = torch.zeros_like(param)

    for inputs, targets in dataloader:
        inputs = inputs.to(device)
        if targets is not None:
            targets = targets.to(device)

        model.zero_grad()
        outputs = model(inputs)
        if targets is not None:
            loss = F.mse_loss(outputs, targets.expand_as(outputs))
        else:
            loss = outputs.mean()
        loss.backward()

        for name, param in model.named_parameters():
            if param.grad is not None:
                fisher_dict[name] += param.grad.data ** 2

    n_samples = len(dataloader)
    for name in fisher_dict:
        fisher_dict[name] /= n_samples

    return fisher_dict


class EWCLoss(nn.Module):
    """Elastic Weight Consolidation loss."""

    def __init__(
        self,
        fisher_dict: Dict[str, torch.Tensor],
        old_params: Dict[str, torch.Tensor],
        lambda_ewc: float = 1000.0,
    ):
        super().__init__()
        self.fisher_dict = fisher_dict
        self.old_params = old_params
        self.lambda_ewc = lambda_ewc

    def forward(self, model: nn.Module) -> torch.Tensor:
        loss = 0.0
        for name, param in model.named_parameters():
            if name in self.fisher_dict and name in self.old_params:
                fisher = self.fisher_dict[name]
                old_param = self.old_params[name]
                loss += (fisher * (param - old_param) ** 2).sum()
        return self.lambda_ewc * loss


class FineTuningPipeline:
    """Complete fine-tuning pipeline for voice profile adaptation."""

    def __init__(
        self,
        base_model: nn.Module,
        output_dir: Path,
        require_gpu: bool = False,
    ):
        if require_gpu and not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for fine-tuning.")

        self.base_model = base_model
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._prior_fisher = None
        self._prior_params = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.base_model.to(self.device)

    def set_prior_knowledge(self, samples: List[Any]) -> None:
        """Establish prior knowledge for EWC from samples."""
        dataloader = [
            (s.mel_tensor.to(self.device), s.speaker_embedding.to(self.device))
            for s in samples
        ]
        self._prior_fisher = compute_fisher_information(self.base_model, dataloader, self.device)
        self._prior_params = {
            name: param.clone().detach()
            for name, param in self.base_model.named_parameters()
        }
        logger.info(f"Computed prior knowledge from {len(samples)} samples")

    def fine_tune(
        self,
        samples: List[Any],
        config: "TrainingConfig",
        mode: str = "lora",
        progress_callback: Optional[Callable[[int, int, float, int], None]] = None,
    ) -> Dict[str, Any]:
        """Execute fine-tuning."""
        from auto_voice.training.job_manager import TrainingConfig

        if mode == "lora":
            model = inject_lora_adapters(
                self.base_model,
                target_modules=config.lora_target_modules,
                rank=config.lora_rank,
                alpha=config.lora_alpha,
                dropout=config.lora_dropout,
            )
            # Note: LoRALinear already freezes original weights internally
        else:
            model = self.base_model
            unfreeze_model(model)

        model.to(self.device)
        model.train()

        dataloader = [
            (s.mel_tensor.to(self.device), s.speaker_embedding.to(self.device))
            for s in samples
        ]

        trainable_params = [p for p in model.parameters() if p.requires_grad]
        if mode == "lora" and hasattr(model, "lora_adapters"):
            trainable_params = list(model.lora_adapters.parameters())

        optimizer = AdamW(trainable_params, lr=config.learning_rate)
        scheduler = CosineAnnealingLR(optimizer, T_max=config.epochs)

        ewc_loss_fn = None
        if config.use_ewc and self._prior_fisher is not None:
            ewc_loss_fn = EWCLoss(self._prior_fisher, self._prior_params, config.ewc_lambda)

        loss_curve = []
        initial_loss = None
        total_steps = config.epochs * len(dataloader)
        current_step = 0
        ewc_term = torch.tensor(0.0, device=self.device)

        for epoch in range(config.epochs):
            epoch_loss = 0.0
            for batch_idx, (inputs, targets) in enumerate(dataloader):
                optimizer.zero_grad()
                outputs = model(inputs)
                task_loss = F.mse_loss(outputs, targets.expand_as(outputs))
                
                ewc_term = torch.tensor(0.0, device=self.device)
                if ewc_loss_fn is not None:
                    ewc_term = ewc_loss_fn(model)

                loss = task_loss + ewc_term
                loss.backward()
                torch.nn.utils.clip_grad_norm_(trainable_params, config.max_grad_norm)
                optimizer.step()

                epoch_loss += loss.item()
                current_step += 1
                if initial_loss is None:
                    initial_loss = loss.item()

                if progress_callback is not None:
                    progress_pct = int(100 * current_step / total_steps)
                    progress_callback(epoch, batch_idx, loss.item(), progress_pct)

            scheduler.step()
            avg_epoch_loss = epoch_loss / len(dataloader)
            loss_curve.append(avg_epoch_loss)
            logger.debug(f"Epoch {epoch+1}/{config.epochs}, Loss: {avg_epoch_loss:.4f}")

        if progress_callback is not None:
            progress_callback(config.epochs - 1, len(dataloader) - 1, loss_curve[-1], 100)

        if mode == "lora":
            adapter_path = self.output_dir / "adapter.pt"
            save_lora_adapter(model, adapter_path)
            result_path_key = "adapter_path"
            result_path = str(adapter_path)
        else:
            model_path = self.output_dir / "model.pt"
            torch.save(model.state_dict(), model_path)
            result_path_key = "model_path"
            result_path = str(model_path)

        return {
            "success": True,
            result_path_key: result_path,
            "initial_loss": initial_loss,
            "final_loss": loss_curve[-1],
            "loss_curve": loss_curve,
            "metrics": {"ewc_loss": ewc_term.item() if ewc_loss_fn else 0.0},
        }


def save_lora_adapter(model: nn.Module, path: Path) -> None:
    """Save LoRA adapter weights and metadata."""
    if not hasattr(model, "lora_adapters"):
        raise ValueError("Model does not have LoRA adapters")

    adapter_states = {}
    legacy_adapters = {}
    for name, adapter in model.lora_adapters.items():
        lora_a = adapter.lora_A.data.cpu()
        lora_b = adapter.lora_B.data.cpu()
        adapter_states[f"{name}.lora_A"] = lora_a
        adapter_states[f"{name}.lora_B"] = lora_b
        legacy_adapters[name] = {
            "lora_A": lora_a,
            "lora_B": lora_b,
            "rank": getattr(adapter, "rank", None),
            "alpha": getattr(adapter, "alpha", None),
        }

    save_dict = build_lora_checkpoint_payload(
        adapter_states,
        config=getattr(model, "_lora_config", {}),
        metadata={
            "adapter_names": list(model.lora_adapters.keys()),
        },
    )
    # Preserve the legacy nested view for profile-storage and old test/runtime
    # callers while keeping the canonical flat tensor keys as the source of truth.
    save_dict["adapters"] = legacy_adapters
    save_dict["config"] = getattr(model, "_lora_config", {})
    torch.save(save_dict, path)
    logger.info(f"Saved LoRA adapter to {path}")


def load_lora_adapter(model: nn.Module, path: Path) -> nn.Module:
    """Load LoRA adapter weights into a model."""
    save_dict = torch.load(path, map_location="cpu")
    metadata = extract_lora_metadata(save_dict)
    config = dict(metadata.get("config", {}))

    model = inject_lora_adapters(
        model,
        target_modules=config.get("target_modules", []),
        rank=config.get("rank", 8),
        alpha=config.get("alpha", 16),
        dropout=config.get("dropout", 0.0),
    )

    adapter_states = extract_lora_state_dict(save_dict)
    for name, adapter in model.lora_adapters.items():
        lora_a = adapter_states.get(f"{name}.lora_A")
        lora_b = adapter_states.get(f"{name}.lora_B")
        if lora_a is not None:
            adapter.lora_A.data.copy_(lora_a)
        if lora_b is not None:
            adapter.lora_B.data.copy_(lora_b)

    logger.info(f"Loaded LoRA adapter from {path}")
    return model


def load_adapter_metadata(path: Path) -> Dict[str, Any]:
    """Load adapter metadata without full weights."""
    save_dict = torch.load(path, map_location="cpu")
    metadata = extract_lora_metadata(save_dict)
    config = dict(metadata.get("config", {}))
    return config


def compute_speaker_similarity(reference: torch.Tensor, generated: torch.Tensor) -> float:
    """Compute cosine similarity between speaker embeddings."""
    reference = reference.flatten()
    generated = generated.flatten()
    similarity = F.cosine_similarity(reference.unsqueeze(0), generated.unsqueeze(0))
    return similarity.item()
