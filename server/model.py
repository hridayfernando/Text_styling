import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import  UNet2DConditionModel, AutoencoderKL, DDPMScheduler
from diffusers import StableDiffusionImg2ImgPipeline
from transformers import CLIPTextModel, CLIPTokenizer
from PIL import Image
import random
import logging
from modelConfig import TrainingConfig
import torchvision.models as models
from typing import List, Dict, Union
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
class LoRALinear(nn.Module):
    """Fixed LoRA implementation with proper dtype handling"""
    
    def __init__(self, original_layer: nn.Linear, rank: int = 64, alpha: float = 64):
        super().__init__()
        self.original_layer = original_layer
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        
        # Freeze original layer
        for param in self.original_layer.parameters():
            param.requires_grad = False
        
        # Initialize with same dtype and device as original layer
        device = original_layer.weight.device
        dtype = original_layer.weight.dtype
        
        self.lora_A = nn.Parameter(
            torch.randn(rank, original_layer.in_features, device=device, dtype=dtype) * 0.01
        )
        self.lora_B = nn.Parameter(
            torch.zeros(original_layer.out_features, rank, device=device, dtype=dtype)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        original_out = self.original_layer(x)
        
        # Ensure dtype consistency
        if x.dtype != self.lora_A.dtype:
            x_lora = x.to(self.lora_A.dtype)
        else:
            x_lora = x
        
        lora_out = F.linear(F.linear(x_lora, self.lora_A), self.lora_B) * self.scaling
        
        # Ensure output dtype matches original
        if lora_out.dtype != original_out.dtype:
            lora_out = lora_out.to(original_out.dtype)
        
        return original_out + lora_out
    
    def get_lora_state_dict(self) -> Dict[str, torch.Tensor]:
        return {
            'lora_A': self.lora_A,
            'lora_B': self.lora_B,
            'scaling': torch.tensor(self.scaling, dtype=self.lora_A.dtype)
        }
class EnhancedStyleLoss(nn.Module):
    """Enhanced loss function combining diffusion loss with perceptual loss"""
    
    def __init__(self, device="cuda", perceptual_weight=0.1):
        super().__init__()
        # Load VGG for perceptual loss
        vgg = models.vgg19(weights='VGG19_Weights.DEFAULT').features.to(device).eval()
        for param in vgg.parameters():
            param.requires_grad = False
        
        self.vgg = vgg
        self.vgg_layers = [2, 7, 12, 21]  # relu1_2, relu2_2, relu3_3, relu4_2
        self.mse_loss = nn.MSELoss()
        self.perceptual_weight = perceptual_weight
        
    def get_vgg_features(self, x):
        """Extract VGG features from multiple layers"""
        features = []
        for i, layer in enumerate(self.vgg):
            x = layer(x)
            if i in self.vgg_layers:
                features.append(x)
        return features
    
    def compute_perceptual_loss(self, pred_image, target_image):
        """Compute perceptual loss using VGG features"""
        # Normalize images to VGG expected range [0,1]
        pred_norm = (pred_image + 1) / 2
        target_norm = (target_image + 1) / 2
        
        pred_features = self.get_vgg_features(pred_norm)
        target_features = self.get_vgg_features(target_norm)
        
        perceptual_loss = 0
        for pred_feat, target_feat in zip(pred_features, target_features):
            perceptual_loss += self.mse_loss(pred_feat, target_feat)
        
        return perceptual_loss
    
    def forward(self, noise_pred, noise_target, pred_image=None, target_image=None):
        """Combined loss function"""
        # Standard diffusion loss
        diffusion_loss = self.mse_loss(noise_pred, noise_target)
        
        # Add perceptual loss if images are provided
        if pred_image is not None and target_image is not None:
            perceptual_loss = self.compute_perceptual_loss(pred_image, target_image)
            total_loss = diffusion_loss + self.perceptual_weight * perceptual_loss
            return total_loss, diffusion_loss, perceptual_loss
        else:
            return diffusion_loss, diffusion_loss, 0.0
        

class DiffusionStyleTransferPipeline:
    """Production-ready diffusion pipeline with memory management"""
    
    def __init__(
        self,
        config: TrainingConfig,
        device: str = "cuda",
        torch_dtype: torch.dtype = torch.float16,
    ):
        self.config = config
        self.device = device
        self.torch_dtype = torch_dtype
        self.lora_layers = {}
        
        logger.info("Initializing Stable Diffusion components...")
        
        # Load components
        self._load_components()
        
        # Setup enhanced loss if enabled
        if config.use_enhanced_loss:
            self.enhanced_loss = EnhancedStyleLoss(device=device, perceptual_weight=config.perceptual_loss_weight)
        
        # Setup inference pipeline (lazy loading)
        self.inference_pipeline = None
        
        logger.info(f"Pipeline initialized successfully")
    
    def _load_components(self):
        """Load SD components with proper memory management"""
        try:
            # Load components
            self.vae = AutoencoderKL.from_pretrained(
                self.config.model_id, 
                subfolder="vae", 
                torch_dtype=self.torch_dtype
            )
            self.text_encoder = CLIPTextModel.from_pretrained(
                self.config.model_id, 
                subfolder="text_encoder", 
                torch_dtype=self.torch_dtype
            )
            self.tokenizer = CLIPTokenizer.from_pretrained(
                self.config.model_id, 
                subfolder="tokenizer"
            )
            self.unet = UNet2DConditionModel.from_pretrained(
                self.config.model_id, 
                subfolder="unet", 
                torch_dtype=self.torch_dtype
            )
            self.scheduler = DDPMScheduler.from_pretrained(
                self.config.model_id, 
                subfolder="scheduler"
            )
            
            # Move to device
            self.vae.to(self.device)
            self.unet.to(self.device)
            
            # Keep text encoder on CPU initially to save memory
            if self.device == "cuda":
                self.text_encoder.to("cpu")
            else:
                self.text_encoder.to(self.device)
            
            # Freeze base models
            self.vae.requires_grad_(False)
            self.text_encoder.requires_grad_(False)
            self.unet.requires_grad_(False)
            
            # Add LoRA to UNet
            self._add_lora_to_unet()
            
        except Exception as e:
            logger.error(f"Error loading components: {e}")
            raise
    
    def _add_lora_to_unet(self):
        """Add LoRA layers with proper module replacement"""
        lora_count = 0
        
        for name, module in self.unet.named_modules():
            if isinstance(module, nn.Linear) and any(target in name for target in ['attn', 'to_q', 'to_k', 'to_v', 'to_out']):
                # Create LoRA layer
                lora_layer = LoRALinear(
                    module, 
                    rank=self.config.lora_rank, 
                    alpha=self.config.lora_alpha
                )
                
                # Ensure proper device and dtype
                lora_layer.to(device=self.device, dtype=self.torch_dtype)
                
                # Replace module
                parent_name = '.'.join(name.split('.')[:-1])
                child_name = name.split('.')[-1]
                
                if parent_name:
                    parent_module = dict(self.unet.named_modules())[parent_name]
                    setattr(parent_module, child_name, lora_layer)
                else:
                    setattr(self.unet, child_name, lora_layer)
                
                self.lora_layers[name] = lora_layer
                lora_count += 1
        
        logger.info(f"Added {lora_count} LoRA layers to UNet")
    
    def encode_text(self, prompts: List[str]) -> torch.Tensor:
        """Encode text prompts with memory management"""
        # Move text encoder to device temporarily
        self.text_encoder.to(self.device)
        
        try:
            text_inputs = self.tokenizer(
                prompts,
                padding="max_length",
                max_length=self.tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            )
            
            with torch.no_grad():
                text_embeddings = self.text_encoder(text_inputs.input_ids.to(self.device))[0]
            
            return text_embeddings
            
        finally:
            # Move back to CPU to save memory
            if self.device == "cuda":
                self.text_encoder.to("cpu")
    
    def create_enhanced_style_prompts(self, style_names: List[str]) -> List[str]:
        """Generate enhanced style transfer prompts"""
        ghibli_templates = [
            "studio ghibli style, anime artwork, soft watercolor painting, hand-drawn animation, whimsical, detailed backgrounds",
            "miyazaki style illustration, ghibli anime, pastel colors, dreamy atmosphere, nature-inspired, magical realism",
            "ghibli movie style, traditional animation, soft lighting, organic shapes, peaceful scenery, artistic masterpiece",
            "spirited away style, howl's moving castle aesthetic, anime art, vibrant yet soft colors, detailed character design"
        ]
        
        general_templates = [
            "{} style artwork, high quality, detailed, masterpiece",
            "beautiful {} style painting, artistic, vibrant colors",
            "{} art style, professional illustration, stunning",
            "amazing {} style image, creative, high resolution",
        ]
        
        prompts = []
        for style_name in style_names:
            if "ghibli" in style_name.lower():
                template = random.choice(ghibli_templates)
                prompts.append(template)
            else:
                template = random.choice(general_templates)
                prompts.append(template.format(style_name))
        
        return prompts
    
    def training_step(
        self,
        original_images: torch.Tensor,
        styled_images: torch.Tensor,
        style_prompts: List[str],
    ) -> Dict[str, torch.Tensor]:
        """Enhanced training step with optional perceptual loss"""
        try:
            batch_size = original_images.shape[0]
            
            # Encode images to latent space
            with torch.no_grad():
                styled_latents = self.vae.encode(styled_images).latent_dist.sample()
                styled_latents *= self.vae.config.scaling_factor
            
            # Sample noise and timesteps
            noise = torch.randn_like(styled_latents)
            timesteps = torch.randint(
                0, self.scheduler.config.num_train_timesteps, 
                (batch_size,), device=self.device
            ).long()
            
            # Add noise to styled images
            noisy_latents = self.scheduler.add_noise(styled_latents, noise, timesteps)
            
            # Encode text prompts
            text_embeddings = self.encode_text(style_prompts)
            
            # Predict noise
            noise_pred = self.unet(
                noisy_latents,
                timesteps,
                encoder_hidden_states=text_embeddings,
            ).sample
            
            # Calculate loss
            if self.config.use_enhanced_loss and hasattr(self, 'enhanced_loss'):
                total_loss, diffusion_loss, perceptual_loss = self.enhanced_loss(
                    noise_pred, noise, styled_images, styled_images
                )
                return {
                    "loss": total_loss,
                    "diffusion_loss": diffusion_loss,
                    "perceptual_loss": perceptual_loss,
                    "noise_pred": noise_pred,
                    "noise": noise,
                }
            else:
                loss = F.mse_loss(noise_pred, noise, reduction="mean")
                return {
                    "loss": loss,
                    "diffusion_loss": loss,
                    "perceptual_loss": 0.0,
                    "noise_pred": noise_pred,
                    "noise": noise,
                }
            
        except torch.cuda.OutOfMemoryError:
            logger.error("CUDA out of memory during training step")
            torch.cuda.empty_cache()
            raise
        except Exception as e:
            logger.error(f"Error in training step: {e}")
            raise
    
    def setup_inference_pipeline(self):
        """Setup inference pipeline with proper safety checker handling"""
        if self.inference_pipeline is None:
            try:
                # Move text encoder back to device for inference
                self.text_encoder.to(self.device)
                
                self.inference_pipeline = StableDiffusionImg2ImgPipeline(
                    vae=self.vae,
                    text_encoder=self.text_encoder,
                    tokenizer=self.tokenizer,
                    unet=self.unet,
                    scheduler=self.scheduler,
                    safety_checker=None,
                    requires_safety_checker=False,  # Fixed safety checker warning
                    feature_extractor=None,
                )
                self.inference_pipeline.to(self.device)
                logger.info("Inference pipeline setup complete")
                
            except Exception as e:
                logger.error(f"Error setting up inference pipeline: {e}")
                raise
    
    def stylize_image(
        self,
        image: Union[str, Image.Image],
        prompt: str,
        strength: float = 0.8,  # Higher strength for better style transfer
        guidance_scale: float = 12.0,  # Higher guidance for better prompt adherence
        num_inference_steps: int = 75,  # More steps for better quality
        negative_prompt: str = "blurry, low quality, distorted, ugly, bad anatomy"
    ) -> Image.Image:
        """Apply learned style to an image with enhanced parameters"""
        try:
            self.setup_inference_pipeline()
            
            if isinstance(image, str):
                with Image.open(image) as img:
                    image = img.convert('RGB').copy()
            
            # Resize image
            image = image.resize((self.config.image_size, self.config.image_size), Image.LANCZOS)
            
            # Generate styled image
            with torch.no_grad():
                result = self.inference_pipeline(
                    prompt=prompt,
                    image=image,
                    strength=strength,
                    guidance_scale=guidance_scale,
                    num_inference_steps=num_inference_steps,
                    negative_prompt=negative_prompt,
                )
            
            return result.images[0]
            
        except Exception as e:
            logger.error(f"Error during image stylization: {e}")
            raise