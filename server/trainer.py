import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import StableDiffusionPipeline, UNet2DConditionModel, AutoencoderKL, DDPMScheduler
from diffusers import StableDiffusionImg2ImgPipeline
from transformers import CLIPTextModel, CLIPTokenizer
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from pathlib import Path
from PIL import Image
import random
import os
import gc
import logging
from tqdm import tqdm
from dataclasses import dataclass
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import torchvision.models as models
from typing import List, Dict, Union, Tuple
from model import DiffusionStyleTransferPipeline
from modelConfig import TrainingConfig
from dataset import StyleTransferDataset

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class StyleTransferTrainer:
    """Production-ready trainer with all improvements"""
    
    def __init__(self, pipeline: DiffusionStyleTransferPipeline, config: TrainingConfig):
        self.pipeline = pipeline
        self.config = config
        
        # Setup optimizer for LoRA parameters only
        self._setup_optimizer()
        
        # Setup mixed precision with updated API
        self.scaler = torch.amp.GradScaler('cuda') if config.mixed_precision and torch.cuda.is_available() else None
        
        # Training metrics
        self.train_losses = []
        self.val_losses = []
        
        logger.info(f"Trainer initialized with {len(self._get_lora_params())} LoRA parameters")
    
    def _get_lora_params(self) -> List[torch.nn.Parameter]:
        """Get all LoRA parameters"""
        lora_params = []
        for lora_layer in self.pipeline.lora_layers.values():
            lora_params.extend(lora_layer.parameters())
        return lora_params
    
    def _setup_optimizer(self):
        """Setup optimizer with learning rate scheduling"""
        lora_params = self._get_lora_params()
        
        self.optimizer = optim.AdamW(
            lora_params,
            lr=self.config.learning_rate,
            weight_decay=0.01,
            betas=(0.9, 0.999)
        )
        
        # Learning rate scheduler
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=self.config.num_epochs,
            eta_min=self.config.learning_rate * 0.1
        )
    
    def train_epoch(self, dataloader: DataLoader, epoch: int) -> float:
        """Train for one epoch with gradient accumulation and enhanced loss logging"""
        total_loss = 0
        total_diffusion_loss = 0
        total_perceptual_loss = 0
        num_batches = len(dataloader)
        
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch}")
        
        self.optimizer.zero_grad()
        
        for batch_idx, (original_images, styled_images, batch_style_names) in enumerate(progress_bar):
            try:
                # Move to device
                original_images = original_images.to(self.pipeline.device)
                styled_images = styled_images.to(self.pipeline.device)
                
                # Create enhanced prompts
                prompts = self.pipeline.create_enhanced_style_prompts(batch_style_names)
                
                # Training step with mixed precision (updated API)
                if self.scaler:
                    with torch.amp.autocast('cuda'):
                        outputs = self.pipeline.training_step(
                            original_images=original_images,
                            styled_images=styled_images,
                            style_prompts=prompts,
                        )
                        loss = outputs["loss"] / self.config.accumulation_steps
                    
                    self.scaler.scale(loss).backward()
                else:
                    outputs = self.pipeline.training_step(
                        original_images=original_images,
                        styled_images=styled_images,
                        style_prompts=prompts,
                    )
                    loss = outputs["loss"] / self.config.accumulation_steps
                    loss.backward()
                
                total_loss += loss.item() * self.config.accumulation_steps
                total_diffusion_loss += outputs.get("diffusion_loss", 0)
                total_perceptual_loss += outputs.get("perceptual_loss", 0)
                
                # Gradient accumulation step
                if (batch_idx + 1) % self.config.accumulation_steps == 0:
                    if self.scaler:
                        self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(
                            self._get_lora_params(),
                            self.config.gradient_clip
                        )
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        torch.nn.utils.clip_grad_norm_(
                            self._get_lora_params(),
                            self.config.gradient_clip
                        )
                        self.optimizer.step()
                    
                    self.optimizer.zero_grad()
                
                # Update progress bar with detailed loss info
                progress_bar.set_postfix({
                    'Total': f'{loss.item() * self.config.accumulation_steps:.4f}',
                    'Diffusion': f'{outputs.get("diffusion_loss", 0):.4f}',
                    'Perceptual': f'{outputs.get("perceptual_loss", 0):.4f}',
                    'LR': f'{self.optimizer.param_groups[0]["lr"]:.2e}'
                })
                 
            except torch.cuda.OutOfMemoryError:
                logger.error("CUDA OOM during training - clearing cache")
                torch.cuda.empty_cache()
                self.optimizer.zero_grad()
                continue
            except Exception as e:
                logger.error(f"Error in batch {batch_idx}: {e}")
                continue
        
        avg_loss = total_loss / num_batches
        self.train_losses.append(avg_loss)
        
        # Step scheduler
        self.scheduler.step()
        
        # Log detailed loss information
        logger.info(f"Epoch {epoch} - Total: {avg_loss:.4f}, "
                    f"Diffusion: {total_diffusion_loss/num_batches:.4f}, "
                    f"Perceptual: {total_perceptual_loss/num_batches:.4f}")
        
        return avg_loss
    
    def validate(self, val_dataloader: DataLoader) -> float:
        """Validation loop"""
        total_val_loss = 0
        num_batches = len(val_dataloader)
        
        with torch.no_grad():
            for original_images, styled_images, batch_style_names in val_dataloader:
                try:
                    original_images = original_images.to(self.pipeline.device)
                    styled_images = styled_images.to(self.pipeline.device)
                    
                    prompts = self.pipeline.create_enhanced_style_prompts(batch_style_names)
                    
                    outputs = self.pipeline.training_step(
                        original_images=original_images,
                        styled_images=styled_images,
                        style_prompts=prompts,
                    )
                    
                    total_val_loss += outputs["loss"].item()
                    
                except Exception as e:
                    logger.warning(f"Error in validation batch: {e}")
                    continue
        
        avg_val_loss = total_val_loss / num_batches if num_batches > 0 else float('inf')
        self.val_losses.append(avg_val_loss)
        
        return avg_val_loss
    
    def save_model(self, path: str, epoch: int = None):
        """Save LoRA weights and training state"""
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            
            # Collect LoRA state dicts
            lora_state_dict = {}
            for name, lora_layer in self.pipeline.lora_layers.items():
                lora_state_dict[name] = lora_layer.get_lora_state_dict()
            
            # Save checkpoint
            checkpoint = {
                'lora_state_dict': lora_state_dict,
                'optimizer_state_dict': self.optimizer.state_dict(),
                'scheduler_state_dict': self.scheduler.state_dict(),
                'config': self.config,
                'epoch': epoch,
                'train_losses': self.train_losses,
                'val_losses': self.val_losses,
            }
            
            torch.save(checkpoint, path)
            logger.info(f"Model saved to {path}")
            
        except Exception as e:
            logger.error(f"Error saving model: {e}")
            raise
    
    def load_model(self, path: str):
        """Load LoRA weights and training state"""
        try:
            checkpoint = torch.load(path, map_location=self.pipeline.device)
            
            # Load LoRA weights
            for name, lora_layer in self.pipeline.lora_layers.items():
                if name in checkpoint['lora_state_dict']:
                    state_dict = checkpoint['lora_state_dict'][name]
                    lora_layer.lora_A.data = state_dict['lora_A']
                    lora_layer.lora_B.data = state_dict['lora_B']
                    lora_layer.scaling = state_dict['scaling'].item()
            
            # Load optimizer and scheduler
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            
            # Load training history
            self.train_losses = checkpoint.get('train_losses', [])
            self.val_losses = checkpoint.get('val_losses', [])
            
            logger.info(f"Model loaded from {path}")
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise

def create_dataloaders(
    original_dir: str, 
    styled_dir: str, 
    style_name: str, 
    config: TrainingConfig
) -> Tuple[DataLoader, DataLoader]:
    """Create train and validation dataloaders"""
    
    # Create full dataset
    full_dataset = StyleTransferDataset(
        original_dir=original_dir,
        styled_dir=styled_dir,
        style_name=style_name,
        image_size=config.image_size,
        augment=True,
        validate_pairs=True,
    )
    
    # Split dataset
    train_size = int((1 - config.validation_split) * len(full_dataset))
    val_size = len(full_dataset) - train_size
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size]
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        drop_last=True,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
        drop_last=False,
    )
    
    return train_loader, val_loader

def enhanced_test_inference(pipeline, style_name, original_dir, output_dir, num_test_images=3):
    """Enhanced test inference with multiple images and better handling"""
    logger.info("Testing inference...")
    
    # Find test images
    test_files = list(Path(original_dir).glob('*.[jp][pn]g'))
    test_files.extend(list(Path(original_dir).glob('*.[JP][PN]G')))
    
    if not test_files:
        logger.warning("No test images found in original directory")
        return
    
    # Test with multiple images (or just the first few)
    num_tests = min(num_test_images, len(test_files))
    
    for i in range(num_tests):
        test_image_path = test_files[i]  # Fixed: use index instead of entire list
        logger.info(f"Testing with image {i+1}/{num_tests}: {test_image_path.name}")
        
        try:
            styled_result = pipeline.stylize_image(
                image=str(test_image_path),
                prompt=f"studio ghibli style, anime artwork, soft watercolor painting, detailed, masterpiece",
                strength=0.8,
                guidance_scale=12.0,
                num_inference_steps=75
            )
            
            # Save with descriptive filename
            test_output_path = f"{output_dir}/test_result_{i+1}_{test_image_path.stem}.png"
            styled_result.save(test_output_path)
            logger.info(f"Test result {i+1} saved to {test_output_path}")
            
        except Exception as e:
            logger.error(f"Error processing test image {test_image_path.name}: {e}")
            continue
    
    logger.info(f"Test inference completed. Results saved in {output_dir}")
