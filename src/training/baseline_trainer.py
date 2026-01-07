import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_
from typing import Dict
from tqdm import tqdm

from .trainer import BaseTrainer
from ..evaluation.metrics import compute_squad_metrics

class BaselineTrainer(BaseTrainer):
    """
    Baseline Trainer: Full fine-tuning of standard GPT-2 (FP32).
    No quantization schemes, no partial freezing.
    """

    def __init__(self, config, model, train_loader, val_loader, tokenizer):
        super().__init__(config, model, train_loader, val_loader, tokenizer)
        self.logger.info("Initializing Baseline Trainer (FP32 Full Fine-Tuning)")

    def setup_optimizer(self) -> optim.Optimizer:
        """
        Create AdamW optimizer for FULL model parameters.
        Overrides BaseTrainer logic which freezes non-LoRA weights.
        """
        self.logger.info("Setting up optimizer for FULL model fine-tuning...")
        
        # Function to separate decay/no-decay params
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {
                'params': [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
                'weight_decay': self.config['training']['weight_decay']
            },
            {
                'params': [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
                'weight_decay': 0.0
            }
        ]

        # Enable gradients for all parameters explicitly
        for param in self.model.parameters():
            param.requires_grad = True
            
        trainable_count = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        self.logger.info(f"Trainable parameters: {trainable_count:,}")

        optimizer = optim.AdamW(
            optimizer_grouped_parameters,
            lr=self.config['training']['learning_rate'],
            betas=(
                self.config['training']['adam_beta1'],
                self.config['training']['adam_beta2']
            ),
            eps=self.config['training']['adam_epsilon']
        )

        return optimizer

    def train_step(self, batch: Dict[str, torch.Tensor]) -> float:
        """
        Single training step for baseline model.
        """
        self.model.train()

        # Forward pass
        outputs = self.model(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask'],
            start_positions=batch['start_positions'],
            end_positions=batch['end_positions']
        )

        loss = outputs.loss
        
        # Backprop
        loss.backward()

        # Gradient clipping
        clip_grad_norm_(
            self.model.parameters(),
            self.config['training']['max_grad_norm']
        )

        # Optimizer step
        self.optimizer.step()
        self.scheduler.step()
        self.optimizer.zero_grad()

        return loss.item()

    def validation_step(self) -> Dict[str, float]:
        """
        Run validation for num_validation_steps.
        """
        self.model.eval()

        total_loss = 0.0
        total_em = 0.0
        total_f1 = 0.0
        
        val_iter = iter(self.val_loader)
        
        # Progress bar matching joint trainer
        val_range = tqdm(
            range(self.config['training']['num_validation_steps']),
            desc="Validation",
            dynamic_ncols=True,
            leave=False
        )

        sample_logs = []

        with torch.no_grad():
            for step_i in val_range:
                try:
                    batch = next(val_iter)
                except StopIteration:
                    val_iter = iter(self.val_loader)
                    batch = next(val_iter)

                # Move to device safely
                batch = {
                    k: v.to(self.config['training']['device']) if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()
                }

                outputs = self.model(
                    input_ids=batch['input_ids'],
                    attention_mask=batch['attention_mask']
                )

                # Compute metrics
                # Note: We compute loss manually since we didn't pass labels to forward() during val
                # consistent with joint trainer style
                
                # Check metrics.py compute_squad_metrics expectations
                start_loss = F.cross_entropy(outputs.start_logits, batch['start_positions'])
                end_loss = F.cross_entropy(outputs.end_logits, batch['end_positions'])
                loss = (start_loss + end_loss) / 2
                
                squad_metrics = compute_squad_metrics(
                    outputs.start_logits,
                    outputs.end_logits,
                    batch,
                    self.tokenizer
                )
                
                total_loss += loss.item()
                total_em += squad_metrics['em']
                total_f1 += squad_metrics['f1']

                # Collect sample logs from first batch
                if step_i < 4:
                    sample_logs.extend(squad_metrics.get('details', [])[:])

        # Average metrics
        steps = self.config['training']['num_validation_steps']
        metrics = {
            'loss': total_loss / steps,
            'em': total_em / steps,
            'f1': total_f1 / steps
        }
        
        # Log samples
        if sample_logs:
            self.logger.info("\n" + "="*50)
            self.logger.info("🔍 VALIDATION PREDICTION SAMPLES (BASELINE)")
            self.logger.info("="*50)
            for i, s in enumerate(sample_logs):
                self.logger.info(f"[{i+1}] Prediction: '{s['prediction']}'")
                gt = s['ground_truth']
                if isinstance(gt, list) and len(gt) == 1 and gt[0] == "":
                    gt_display = "<Unanswerable>"
                else:
                    gt_display = str(gt)
                self.logger.info(f"    Ground Truth: {gt_display}")
                self.logger.info(f"    Metrics: EM={s['em']:.1f}, F1={s['f1']:.2f}")
                self.logger.info("-" * 30)

        # Prefix for consistency with logger
        return {f"aggregate_{k}": v for k, v in metrics.items()}

    def train(self):
        """Main training loop."""
        pbar = tqdm(
            total=self.config['training']['num_steps'],
            desc="Training (Baseline)",
            dynamic_ncols=True
        )

        for step in range(1, self.config['training']['num_steps'] + 1):
            batch = self.get_next_batch()
            
            loss = self.train_step(batch)
            
            pbar.update(1)
            pbar.set_postfix({
                'loss': f"{loss:.4f}",
                'lr': f"{self.scheduler.get_last_lr()[0]:.2e}"
            })

            # Logging
            if step % self.config['training']['log_every_steps'] == 0:
                self.log_metrics({'total_loss': loss}, step, phase='train')

            # Validation
            if step % self.config['training']['validate_every_steps'] == 0:
                self.logger.info(f"Running validation at step {step}...")
                val_metrics = self.validation_step()
                self.log_metrics(val_metrics, step, phase='val')

                self.logger.info(
                    f"Validation: "
                    f"Loss={val_metrics['aggregate_loss']:.4f}, "
                    f"EM={val_metrics['aggregate_em']:.4f}, "
                    f"F1={val_metrics['aggregate_f1']:.4f}"
                )

        pbar.close()
        
        if self.config['checkpointing']['save_final']:
            self.save_checkpoint(
                step=self.config['training']['num_steps'],
                metrics={'final': True}
            )
            self.logger.info("Baseline training complete!")
