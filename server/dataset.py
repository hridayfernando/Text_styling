import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from pathlib import Path
from PIL import Image
import random
import logging
from typing import List, Dict, Tuple

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class StyleTransferDataset(Dataset):
    """Robust dataset implementation with validation and error handling"""
    
    def __init__(
        self,
        original_dir: str,
        styled_dir: str,
        style_name: str,
        image_size: int = 512,
        augment: bool = True,
        validate_pairs: bool = True,
    ):
        self.original_dir = Path(original_dir)
        self.styled_dir = Path(styled_dir)
        self.image_size = image_size
        self.style_name = style_name
        
        # Validate directories
        self._validate_directories()
        
        # Load and validate image pairs
        self.image_pairs = self._load_and_validate_pairs(validate_pairs)
        
        # Setup transforms
        self.transform = self._setup_transforms(augment)
        
        logger.info(f"Loaded {len(self.image_pairs)} valid image pairs for style: {self.style_name}")
    
    def _validate_directories(self):
        """Validate input directories exist and contain images"""
        if not self.original_dir.exists():
            raise FileNotFoundError(f"Original directory not found: {self.original_dir}")
        if not self.styled_dir.exists():
            raise FileNotFoundError(f"Styled directory not found: {self.styled_dir}")
        
        # Check for image files
        orig_files = list(self.original_dir.glob('*.[jp][pn]g')) + list(self.original_dir.glob('*.[JP][PN]G'))
        style_files = list(self.styled_dir.glob('*.[jp][pn]g')) + list(self.styled_dir.glob('*.[JP][PN]G'))
        
        if not orig_files:
            raise ValueError(f"No image files found in {self.original_dir}")
        if not style_files:
            raise ValueError(f"No image files found in {self.styled_dir}")
    
    def _load_and_validate_pairs(self, validate: bool) -> List[Dict]:
        """Load and validate image pairs with error handling"""
        pairs = []
        
        # Get all image files
        extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.JPG', '*.JPEG', '*.PNG', '*.BMP']
        orig_files = []
        style_files = []
        
        for ext in extensions:
            orig_files.extend(self.original_dir.glob(ext))
            style_files.extend(self.styled_dir.glob(ext))
        
        # Sort for consistent pairing
        orig_files.sort()
        style_files.sort()
        
        # Create and validate pairs
        min_len = min(len(orig_files), len(style_files))
        
        for i in range(min_len):
            orig_path = orig_files[i]
            style_path = style_files[i]
            
            if validate and not self._validate_image_pair(orig_path, style_path):
                logger.warning(f"Skipping invalid pair: {orig_path.name} - {style_path.name}")
                continue
            
            pairs.append({
                'original_path': orig_path,
                'styled_path': style_path,
                'style_name': self.style_name
            })
        
        if not pairs:
            raise ValueError("No valid image pairs found")
        
        return pairs
    
    def _validate_image_pair(self, orig_path: Path, style_path: Path) -> bool:
        """Validate that both images can be loaded and have reasonable dimensions"""
        try:
            with Image.open(orig_path) as orig_img:
                with Image.open(style_path) as style_img:
                    # Check if images are valid and have reasonable size
                    if orig_img.size[0] < 64 or orig_img.size[1] < 64:
                        return False
                    if style_img.size[0] < 64 or style_img.size[1] < 64:
                        return False
                    return True
        except Exception as e:
            logger.warning(f"Error validating pair {orig_path.name}: {e}")
            return False
    
    def _setup_transforms(self, augment: bool) -> transforms.Compose:
        """Setup enhanced image transforms"""
        transform_list = [
            transforms.Resize((self.image_size, self.image_size), interpolation=transforms.InterpolationMode.LANCZOS),
        ]
        
        if augment:
            transform_list.extend([
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=5),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                transforms.RandomAdjustSharpness(sharpness_factor=1.5, p=0.3),
            ])
        
        transform_list.extend([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        
        return transforms.Compose(transform_list)
    
    def __len__(self) -> int:
        return len(self.image_pairs)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, str]:
        """Get a single data sample with error handling"""
        try:
            pair = self.image_pairs[idx]
            
            # Load images
            with Image.open(pair['original_path']) as orig_img:
                original_image = orig_img.convert('RGB').copy()
            
            with Image.open(pair['styled_path']) as style_img:
                styled_image = style_img.convert('RGB').copy()
            
            # Apply transforms
            original_tensor = self.transform(original_image)
            styled_tensor = self.transform(styled_image)
            
            return original_tensor, styled_tensor, pair['style_name']
            
        except Exception as e:
            logger.error(f"Error loading sample {idx}: {e}")
            # Return a random valid sample instead
            return self.__getitem__(random.randint(0, len(self) - 1))