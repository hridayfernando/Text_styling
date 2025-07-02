
import logging
from dataclasses import dataclass


# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class TrainingConfig:
    """Enhanced configuration with validation"""
    model_id: str = "runwayml/stable-diffusion-v1-5"
    image_size: int = 512
    batch_size: int = 2
    effective_batch_size: int = 8
    learning_rate: float = 1e-5
    num_epochs: int = 100  # Increased for better training
    save_every: int = 10
    mixed_precision: bool = True
    gradient_clip: float = 1.0
    lora_rank: int = 128  # Increased for more capacity
    lora_alpha: float = 128
    validation_split: float = 0.2
    accumulation_steps: int = 4
    use_enhanced_loss: bool = True
    perceptual_loss_weight: float = 0.1