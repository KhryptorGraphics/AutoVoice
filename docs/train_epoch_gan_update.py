"""
Two-Step GAN Training Implementation for VoiceConversionTrainer.train_epoch()

Replace the existing train_epoch method in src/auto_voice/training/trainer.py
starting at line ~1186 with this implementation.

This adds discriminator updates before generator updates in each training iteration.
"""

def train_epoch(self, dataloader, epoch: int) -> Dict[str, float]:
    """Train for one epoch with voice conversion losses and adversarial training.

    Implements two-step GAN training:
    1. Update discriminator with real and fake audio
    2. Update generator with reconstruction and adversarial losses

    Args:
        dataloader: Training data loader
        epoch: Current epoch number

    Returns:
        Dict with average losses for the epoch
    """
    self.model.train()
    self.discriminator.train()
    epoch_losses = {}
    num_batches = 0

    progress_bar = tqdm(
        dataloader,
        desc=f"Epoch {epoch}",
        disable=(getattr(self.config, 'local_rank', 0) != 0)
    )

    for batch_idx, batch in enumerate(progress_bar):
        # Move batch to device
        batch = self._move_batch_to_device(batch)

        # ========================================
        # STEP 1: Update Discriminator
        # ========================================
        if self.vc_loss_weights.get('adversarial', 0) > 0:
            # Zero discriminator gradients
            self.discriminator_optimizer.zero_grad()

            with autocast(enabled=self.config.use_amp):
                # Forward pass (with detached generator outputs)
                predictions = self._forward_pass(batch)

                # Get real and fake audio
                if 'pred_audio' in predictions and 'target_audio' in batch:
                    real_audio = batch['target_audio']
                    fake_audio = predictions['pred_audio'].detach()  # Detach to prevent generator updates

                    # Forward through discriminator
                    real_logits_list, _ = self.discriminator(real_audio)
                    fake_logits_list, _ = self.discriminator(fake_audio)

                    # Compute discriminator loss
                    disc_loss = self.hinge_discriminator_loss(real_logits_list, fake_logits_list)
                    disc_loss = disc_loss / self.config.gradient_accumulation_steps

            # Backward pass for discriminator
            if self.scaler:
                self.scaler.scale(disc_loss).backward()
            else:
                disc_loss.backward()

            # Discriminator optimizer step (with gradient accumulation)
            if (batch_idx + 1) % self.config.gradient_accumulation_steps == 0:
                if self.scaler:
                    self.scaler.unscale_(self.discriminator_optimizer)

                # Gradient clipping
                if self.config.gradient_clip > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.discriminator.parameters(),
                        self.config.gradient_clip
                    )

                if self.scaler:
                    self.scaler.step(self.discriminator_optimizer)
                else:
                    self.discriminator_optimizer.step()

                self.discriminator_optimizer.zero_grad()

        # ========================================
        # STEP 2: Update Generator
        # ========================================
        # Zero generator gradients
        self.optimizer.zero_grad()

        with autocast(enabled=self.config.use_amp):
            # Forward pass (without detaching)
            predictions = self._forward_pass(batch)

            # Compute all losses (including adversarial)
            losses = self._compute_voice_conversion_losses(predictions, batch)

        # Backward pass for generator
        loss = losses['total'] / self.config.gradient_accumulation_steps

        if self.scaler:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()

        # Generator optimizer step (with gradient accumulation)
        if (batch_idx + 1) % self.config.gradient_accumulation_steps == 0:
            if self.scaler:
                self.scaler.unscale_(self.optimizer)

            # Gradient clipping
            if self.config.gradient_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.gradient_clip
                )

            if self.scaler:
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                self.optimizer.step()

            self.optimizer.zero_grad()

            if self.scheduler:
                self.scheduler.step()

            self.global_step += 1

        # Accumulate losses (including discriminator loss if available)
        for loss_name, loss_value in losses.items():
            if loss_name not in epoch_losses:
                epoch_losses[loss_name] = 0.0
            epoch_losses[loss_name] += loss_value.item()

        if self.vc_loss_weights.get('adversarial', 0) > 0 and 'pred_audio' in predictions:
            if 'disc_loss' not in epoch_losses:
                epoch_losses['disc_loss'] = 0.0
            epoch_losses['disc_loss'] += disc_loss.item() * self.config.gradient_accumulation_steps

        num_batches += 1

        # Update progress bar
        postfix = {
            'G_loss': losses['total'].item(),
            'mel': losses.get('mel_reconstruction', torch.tensor(0)).item(),
            'adv': losses.get('adversarial', torch.tensor(0)).item()
        }
        if self.vc_loss_weights.get('adversarial', 0) > 0:
            postfix['D_loss'] = disc_loss.item() * self.config.gradient_accumulation_steps
        progress_bar.set_postfix(postfix)

        # Log to tensorboard/wandb
        if self.global_step % self.config.log_interval == 0:
            log_losses = losses.copy()
            if self.vc_loss_weights.get('adversarial', 0) > 0 and 'pred_audio' in predictions:
                log_losses['disc_loss'] = disc_loss * self.config.gradient_accumulation_steps
            self._log_training_step(log_losses, losses['total'].item())

    # Average losses
    for loss_name in epoch_losses:
        epoch_losses[loss_name] /= num_batches

    return epoch_losses
