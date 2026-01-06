import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, Optional, Any
import os
import json
import logging
from datetime import datetime
import wandb
from transformers import get_linear_schedule_with_warmup


class BaseTrainer:
    """
    Base trainer class with shared infrastructure.

    Provides:
    - Optimizer setup (AdamW, LoRA-only)
    - LR scheduler (warmup + decay)
    - Logging (file + wandb)
    - Checkpointing
    - Experiment directory management    
    """
    def __init__(
        self,
        config: Dict[str, Any],  # Config dict from YAML
        model,  # QuantizedGPT2ForQuestionAnswering
        train_loader: DataLoader,
        val_loader: DataLoader,
        tokenizer
    ):
        """
        Initialize base trainer.
        
        Args:
            config: Config dict from config_loader (preserves YAML hierarchy)
            model: QuantizedGPT2ForQuestionAnswering instance
            train_loader: Training data loader
            val_loader: Validation data loader
            tokenizer: GPT2TokenizerFast instance
        """
        self.config = config
        self.model = model.to(config['training']['device'])
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.tokenizer = tokenizer
        # Setup logging and experiment directory
        self.experiment_dir = self.create_experiment_dir()

        self.logger = self.setup_file_logger()
        
        # Setup training components
        self.optimizer = self.setup_optimizer()

        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=config['training']['warmup_steps'],
            num_training_steps=config['training']['num_steps']
        )


        # Log initial info
        self.logger.info("=" * 80)
        self.logger.info(f"Experiment: {config['experiment']['name']}")
        self.logger.info(f"Type: {config['experiment']['type']}")
        self.logger.info(f"Device: {config['training']['device']}")
        self.logger.info(f"Experiment directory: {self.experiment_dir}")
        self.logger.info("=" * 80)    

        # Setup wandb
        self.setup_wandb()

        # Training state
        self.step = 0
        self.train_iter = iter(self.train_loader)

    def setup_optimizer(self) -> optim.Optimizer:
        """
        Create AdamW optimizer.
        
        Only optimize LoRA parameters!
        Freeze all base model parameters.
        
        Returns:
            AdamW optimizer
        """       
        # Collect LoRA parameters
        lora_params = []
        frozen_count = 0
        trainable_count = 0

        for name, param in self.model.named_parameters():
            if 'lora' in name:
                param.requires_grad = True
                lora_params.append(param)
                trainable_count += param.numel()
            else:
                param.requires_grad = False
                frozen_count += param.numel()

        self.logger.info(f"Parameters: {trainable_count:,} trainable (LoRA), {frozen_count:,} frozen")

        # Create optimizer
        optimizer = optim.AdamW(
            lora_params,
            lr=self.config['training']['learning_rate'],
            weight_decay=self.config['training']['weight_decay'],
            betas=(
                self.config['training']['adam_beta1'],
                self.config['training']['adam_beta2']
            ),
            eps=self.config['training']['adam_epsilon']
        )

        return optimizer
        
    def create_experiment_dir(self) -> str:
        """
        Create experiment directory structure.
        
        Structure:
            experiments/{experiment_name}_{timestamp}/
            ├── config.yaml (copied later by train.py)
            ├── logs/
            │   └── experiment.log
            └── checkpoints/
        
        Returns:
            Path to experiment directory        
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        exp_name = f"{self.config['experiment']['name']}_{timestamp}"
        exp_dir = os.path.join(self.config['experiment']['base_folder'], exp_name)

        os.makedirs(os.path.join(exp_dir, "logs"), exist_ok=True)
        os.makedirs(os.path.join(exp_dir, "checkpoints"), exist_ok=True)

        return exp_dir


    def setup_file_logger(self) -> logging.Logger:
        """
        Setup file and console logger.

        Returns:
            Logger instance
        """
        logger = logging.getLogger(f"trainer_{self.config['experiment']['name']}_{id(self)}")
        logger.setLevel(logging.INFO)

        # Clear any existing handlers
        logger.handlers = []

        # File handler
        log_file = os.path.join(self.experiment_dir, "logs", "experiment.log")
        fh = logging.FileHandler(log_file)
        fh.setLevel(logging.INFO)

        # Console handler
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)

        # Formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)
        
        logger.addHandler(fh)
        logger.addHandler(ch)

        return logger

    def setup_wandb(self):
        """
        Initialize W&B logging.
        """
        if self.config['logging']['use_wandb']:
            wandb.init(
                project=self.config['logging']['wandb_project'],
                entity=self.config['logging']['wandb_entity'],
                name=f"{self.config['experiment']['name']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                config=self.config  # Pass full config dict
            )         

            self.logger.info("✅ Wandb initialized")


    def compute_efficiency_metrics(
        self,
        config_name: str,
        model_bit_width_config
    ) -> Dict[str, float]:
        """
        Compute theoretical efficiency metrics.
        
        Metrics:
        1. Model size (MB) = sum of (params x bits) / (8 x 2^20)
        2. FLOPs multiplier (relative to FP32)
        3. Speedup estimate
        Args:
            config_name: Name of bit-width config
            model_bit_width_config: ModelBitWidthConfig instance
        
        Returns:
            Dict with efficiency metrics        
        """
        flop_multipliers = {i: i/32.0 for i in range(2, 33)}
        
        total_bits = 0
        total_flops_multiplier = 0.0
        num_layers = 0

        # Iterate through layers
        for layer_idx, layer_config in enumerate(model_bit_width_config.layer_configs):
            # Get parameter count for this layer
            layer_params = self.count_layer_params(layer_idx)

            # Weight bits contribution
            weight_bits = layer_config.bit_width_w
            total_bits += layer_params * weight_bits

            # FLOPs contribution
            total_flops_multiplier += flop_multipliers.get(weight_bits, 1.0)

            num_layers += 1

        # Calculate metrics
        model_size_mb = total_bits / (8 * 2**20)
        avg_flops_multiplier = total_flops_multiplier / num_layers
        speedup_estimate = 1.0 / avg_flops_multiplier

        return {
            'model_size_mb': model_size_mb,
            'flops_multiplier': avg_flops_multiplier,
            'speedup_estimate': speedup_estimate
        }

    def count_layer_params(self, layer_idx: int) -> int:
        """
        Count parameters in a specific transformer layer.
        
        For GPT-2, each layer has:
        - Attention: 4 linear layers (q, k, v, out)
        - MLP: 2 linear layers (c_fc, c_proj)
        
        Args:
            layer_idx: Index of layer (0-11)
        
        Returns:
            Number of parameters in layer
        """
        hidden_size = self.model.config.hidden_size

        # Attention: 4 × (hidden_size × hidden_size)
        # Each of q_proj, k_proj, v_proj, out_proj
        attn_params = 4 * (hidden_size * hidden_size)
        
        # MLP: 
        # - c_fc: hidden_size × (4 * hidden_size)
        # - c_proj: (4 * hidden_size) × hidden_size
        mlp_params = hidden_size * (4 * hidden_size) + (4 * hidden_size) * hidden_size

        return attn_params + mlp_params  

    def save_checkpoint(self, step: int, metrics: Dict[str, float]):
        """
        Save model checkpoint.
        
        Args:
            step: Current training step
            metrics: Metrics to save with checkpoint
        """
        checkpoint_path = os.path.join(
            self.experiment_dir,
            "checkpoints",
            self.config['checkpointing']['checkpoint_name']
        )
        
        checkpoint = {
            'step': step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'metrics': metrics,
            'config': self.config  # Already a dict
        }        

        torch.save(checkpoint, checkpoint_path)
        self.logger.info(f"✅ Checkpoint saved: {checkpoint_path}")

    def log_metrics(
        self,
        metrics: Dict[str, Any],
        step: int,
        phase: str = "train"
    ):
        """
        Log metrics to file and wandb.
        
        Args:
            metrics: Dict of metrics to log
            step: Current training step
            phase: 'train' or 'val'
        """
        # Format for file logging
        metrics_strs = []
        for key, value in metrics.items():
            if isinstance(value, float):
                metrics_strs.append(f"{key}={value:.4f}")
            else:
                metrics_strs.append(f"{key}={value}")

        log_str = f"Step {step} [{phase}]: " + ", ".join(metrics_strs)
        self.logger.info(log_str)

        # Wandb logging
        if self.config['logging']['use_wandb']:
            wandb_metrics = {}
            for key, value in metrics.items():
                if isinstance(value, (int, float)):
                    wandb_metrics[f"{phase}/{key}"] = value
                wandb_metrics['step'] = step
                wandb.log(wandb_metrics, step=step)

    def get_next_batch(self) -> Dict[str, torch.Tensor]:
        """
        Get next batch from train loader with wraparound.
        
        Returns:
            Batch dict with tensors moved to device
        """        
        try:
            batch = next(self.train_iter)
        except StopIteration:
            # restart iterator when exhausted
            self.train_iter = iter(self.train_loader)
            batch = next(self.train_iter)

        # Move all tensors to device
        batch = {
            k: v.to(self.config['training']['device']) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }

        return batch

    def train(self):
        """
        Main train loop.
        
        TO be implemented by subclasses (JointTrainer, CyclicTrainer)
        """
        raise NotImplementedError("Subclasses must implement train() method")        
