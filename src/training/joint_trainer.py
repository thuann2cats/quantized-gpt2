import torch
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from typing import Dict
from tqdm import tqdm

from .trainer import BaseTrainer
from ..evaluation.metrics import compute_squad_metrics


class JointTrainer(BaseTrainer):
    """
    Joint Training: All bit-width configs trained simultaneously on each batch.
    
    Training loop per step:
    1. Get batch
    2. For each config: forward pass + compute loss
    3. Aggregate losses (mean)
    4. Backprop + optimizer step
    """

    def __init__(self, config, model, train_loader, val_loader, tokenizer):
        super().__init__(config, model, train_loader, val_loader, tokenizer)

        # Get bit-width config names from config
        self.bit_width_config_names = list(config['model']['bit_width_configs'].keys())

        self.logger.info(f"Joint training with {len(self.bit_width_config_names)} configs")
        self.logger.info(f"Configs: {self.bit_width_config_names}")

        # Log efficiency metrics once at start
        self.log_efficiency_metrics()

    def log_efficiency_metrics(self):
        """
        Compute and log theoretical efficiency metrics for all configs.
        """
        self.logger.info("=" * 80)
        self.logger.info("Theoretical Efficiency Metrics")
        self.logger.info("=" * 80)

        for config_name in self.bit_width_config_names:
            # Get the ModelBitWidthConfig for this config name
            model_bit_width_config = self.config['model']['bit_width_configs'][config_name]
            metrics = self.compute_efficiency_metrics(config_name, model_bit_width_config)

            self.logger.info(
                f"{config_name}: "
                f"Size={metrics['model_size_mb']:.2f} MB, "
                f"Speedup={metrics['speedup_estimate']:.2f}x"
            )

            if self.config['logging']['use_wandb']:
                import wandb
                for metric_name, value in metrics.items():
                    wandb.run.summary[f"efficiency/{config_name}/{metric_name}"] = value

        self.logger.info("=" * 80)   

    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """
        Single training step for joint training.
        
        Returns:
            Dict with per-config losses and total loss
        """
        self.model.train()

        losses_by_config = {}
        loss_tensors = []

        # Forward pass for ALL configs
        for config_name in self.bit_width_config_names:
            # Set bit-width config
            self.model.set_model_bit_width_config(config_name)

            # Forward pass
            outputs = self.model(
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask']
            )

            # Compute loss (CrossEntropy for start/end positions)
            start_loss = F.cross_entropy(
                outputs.start_logits,
                batch['start_positions'],
                # ignore_index=-1
            )
            end_loss = F.cross_entropy(
                outputs.end_logits,
                batch['end_positions'],
                # ignore_index=-1
            )
            loss = (start_loss + end_loss) / 2

            # Store for logging and aggregation
            losses_by_config[config_name] = loss.item()
            # loss_tensors.append(loss)

            scaled_loss = loss / len(self.bit_width_config_names)
            scaled_loss.backward()

        # Aggregate losses (mean across configs)
        # total_loss = torch.stack(loss_tensors).mean()
        total_loss = sum(losses_by_config.values()) / len(losses_by_config)

        # Backprop
        # total_loss.backward()

        # Gradient clipping
        clip_grad_norm_(
            self.model.parameters(),
            self.config['training']['max_grad_norm']
        )

        # Optimizer step
        self.optimizer.step()
        self.scheduler.step()
        self.optimizer.zero_grad()

        # Return all losses for logging
        result = {'total_loss': total_loss}
        result.update(losses_by_config)

        return result

    def validation_step(self) -> Dict[str, float]:
        """
        Run validation for num_validation_steps.
        
        Returns:
            Dict with metrics per config and aggregate
        """
        self.model.eval()

        # Accumulators for each config
        metrics_by_config = {
            config_name: {'loss': 0.0, 'em': 0.0, 'f1': 0.0}
            for config_name in self.bit_width_config_names
        }

        val_iter = iter(self.val_loader)
        
        # Progress bar for validation
        val_range = tqdm(
            range(self.config['training']['num_validation_steps']),
            desc="Validation",
            dynamic_ncols=True,
            leave=False
        )

        # Store samples for logging
        sample_logs = []

        with torch.no_grad():

            for step_i in val_range:
                try:
                    batch = next(val_iter)
                except StopIteration:
                    val_iter = iter(self.val_loader)
                    batch = next(val_iter)

                # Move to device
                # batch = {k: v.to(self.config['training']['device']) for k, v in batch.items()}
                batch = {
                    k: v.to(self.config['training']['device']) if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()
                }

                # Evaluate each config
                for config_name in self.bit_width_config_names:
                    self.model.set_model_bit_width_config(config_name)

                    outputs = self.model(
                        input_ids=batch['input_ids'],
                        attention_mask=batch['attention_mask']
                    )

                    # Loss
                    start_loss = F.cross_entropy(
                        outputs.start_logits,
                        batch['start_positions'],
                        ignore_index=-1
                    )
                    end_loss = F.cross_entropy(
                        outputs.end_logits,
                        batch['end_positions'],
                        ignore_index=-1
                    )

                    loss = (start_loss + end_loss) / 2

                    # Metrics
                    squad_metrics = compute_squad_metrics(
                        outputs.start_logits,
                        outputs.end_logits,
                        batch,
                        self.tokenizer,
                    )

                    # Accumulate
                    metrics_by_config[config_name]['loss'] += loss.item()
                    metrics_by_config[config_name]['em'] += squad_metrics['em']
                    metrics_by_config[config_name]['f1'] += squad_metrics['f1']

                    # Collect samples from the first batch and first config
                    if step_i == 0 and config_name == self.bit_width_config_names[0]:
                        sample_logs.extend(squad_metrics.get('details', [])[:])


        # Average over validation steps
        for config_name in self.bit_width_config_names:
            metrics_by_config[config_name]['loss'] /= self.config['training']['num_validation_steps']
            metrics_by_config[config_name]['em'] /= self.config['training']['num_validation_steps']
            metrics_by_config[config_name]['f1'] /= self.config['training']['num_validation_steps']

        # Compute aggregate metrics (mean across configs)
        aggregate = {
            'loss': sum(m['loss'] for m in metrics_by_config.values()) / len(self.bit_width_config_names),
            'em': sum(m['em'] for m in metrics_by_config.values()) / len(self.bit_width_config_names),
            'f1': sum(m['f1'] for m in metrics_by_config.values()) / len(self.bit_width_config_names)
        }

        # Flatten for logging
        result = {'aggregate_' + k: v for k, v in aggregate.items()}

        for config_name, metrics in metrics_by_config.items():
            for metric_name, value in metrics.items():
                result[f"{config_name}_{metric_name}"] = value
        
        # Log sampled predictions
        if sample_logs:
            self.logger.info("\n" + "="*50)
            self.logger.info("🔍 VALIDATION PREDICTION SAMPLES")
            self.logger.info("="*50)
            for i, s in enumerate(sample_logs):
                self.logger.info(f"[{i+1}] Prediction: '{s['prediction']}'")
                gt = s['ground_truth']
                # If gt is [""] or similar, make it more readable
                if isinstance(gt, list) and len(gt) == 1 and gt[0] == "":
                    gt_display = "<Unanswerable>"
                else:
                    gt_display = str(gt)
                self.logger.info(f"    Ground Truth: {gt_display}")
                self.logger.info(f"    Metrics: EM={s['em']:.1f}, F1={s['f1']:.2f}")
                self.logger.info("-" * 30)

        return result

    def train(self):
        """
        Main training loop for joint training.
        """

        # Progress bar
        pbar = tqdm(
            total=self.config['training']['num_steps'],
            desc="Training",
            dynamic_ncols=True
        )

        for step in range(1, self.config['training']['num_steps'] + 1):
            # Get batch
            batch = self.get_next_batch()

            # Training step
            train_metrics = self.train_step(batch)
            
            # Update progress bar
            pbar.update(1)
            pbar.set_postfix({
                'loss': f"{train_metrics['total_loss']:.4f}",
                'lr': f"{self.scheduler.get_last_lr()[0]:.2e}"
            })

            # Logging
            if step % self.config['training']['log_every_steps'] == 0:
                self.log_metrics(train_metrics, step, phase='train')

            # Validation
            if step % self.config['training']['validate_every_steps'] == 0:
                self.logger.info(f"Running validation at step {step}...")
                val_metrics = self.validation_step()
                self.log_metrics(val_metrics, step, phase='val')

                # Log aggregate metrics to console
                self.logger.info(
                    f"Validation: "
                    f"Loss={val_metrics['aggregate_loss']:.4f}, "
                    f"EM={val_metrics['aggregate_em']:.4f}, "
                    f"F1={val_metrics['aggregate_f1']:.4f}"
                )

        pbar.close()

        # Save final checkpoint
        if self.config['checkpointing']['save_final']:
            self.logger.info("Saving final checkpoint...")
            self.save_checkpoint(
                step=self.config['training']['num_steps'],
                metrics={'final': True}
            )
            self.logger.info(f"Final checkpoint saved to {self.experiment_dir}/checkpoints/")
        
        self.logger.info("Training complete!")