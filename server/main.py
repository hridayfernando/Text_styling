from flask import Flask, request, jsonify
from PIL import Image
import os
import torch
import time
from model import DiffusionStyleTransferPipeline
from trainer import StyleTransferTrainer
from modelConfig import TrainingConfig
import logging
from trainer import create_dataloaders,enhanced_test_inference
app = Flask(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
UPLOAD_DIR_ORIGINAL = 'uploads/original_images'
UPLOAD_DIR_STYLE = 'uploads/style_images'
os.makedirs(UPLOAD_DIR_ORIGINAL, exist_ok=True)
os.makedirs(UPLOAD_DIR_STYLE,exist_ok=True)

def train_model(style_name):
    """Main training function with all improvements"""
    # Enhanced configuration
    config = TrainingConfig(
        batch_size=2,
        effective_batch_size=8,
        num_epochs=25,  # Increased for better training
        learning_rate=5e-5,  # Slightly higher learning rate
        save_every=10,
        lora_rank=128,  # Higher rank for more capacity
        lora_alpha=128,
        use_enhanced_loss=True,
        perceptual_loss_weight=0.1
    )
    
    # Setup directories - CUSTOMIZE THESE PATHS
    original_dir = "uploads/original_images"  # Path to original images
    styled_dir = "uploads/style_images"      # Path to styled images
    style_name = style_name                        
    output_dir = "outputs"
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Validate directories
    if not os.path.exists(original_dir) or not os.path.exists(styled_dir):
        logger.error("Please update the directory paths in the main() function!")
        logger.error(f"Looking for:")
        logger.error(f"  Original images: {original_dir}")
        logger.error(f"  Styled images: {styled_dir}")
        return
    
    # Setup device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if device == "cuda" else torch.float32
    logger.info(f"Using device: {device} with dtype: {torch_dtype}")
    
    try:
        # Initialize pipeline and trainer
        pipeline = DiffusionStyleTransferPipeline(
            config=config,
            device=device,
            torch_dtype=torch_dtype
        )
        
        trainer = StyleTransferTrainer(pipeline, config)
        
        # Create dataloaders
        train_loader, val_loader = create_dataloaders(
            original_dir, styled_dir, style_name, config
        )
        logger.info(f"Created dataloaders: {len(train_loader)} train, {len(val_loader)} val batches")
        
        # Training loop
        logger.info("Starting training...")
        best_val_loss = float('inf')
        
        for epoch in range(config.num_epochs):
            # Train epoch
            avg_train_loss = trainer.train_epoch(train_loader, epoch)
            
            # Validate
            avg_val_loss = trainer.validate(val_loader)
            
            logger.info(
                f"Epoch {epoch}: Train Loss = {avg_train_loss:.4f}, "
                f"Val Loss = {avg_val_loss:.4f}, "
                f"LR = {trainer.optimizer.param_groups[0]['lr']:.2e}"
            )
            
            # Save best model
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_path = f"{output_dir}/best_model.pt"
                trainer.save_model(best_path, epoch)
            
            # Save periodic checkpoints
            if (epoch + 1) % config.save_every == 0:
                checkpoint_path = f"{output_dir}/checkpoint_epoch_{epoch+1}.pt"
                trainer.save_model(checkpoint_path, epoch)
        
        # Save final model
        final_path = f"{output_dir}/final_model.pt"
        trainer.save_model(final_path, config.num_epochs)
        
        logger.info("Training completed successfully!")
        
        # Enhanced test inference
        enhanced_test_inference(pipeline, style_name, original_dir, output_dir, num_test_images=3)
        
    except Exception as e:
        logger.error(f"Training failed with error: {e}")
        raise



@app.route('/train', methods=['POST'])
def train():
    original_images = request.files.getlist('original_images')
    styled_images = request.files.getlist('styled_images')
    saved_files = []
    if len(original_images) != len(styled_images):
        return jsonify({'status': 'error', 'msg':'lenght mis match between original and styled images'})
    timestamp = int(time.time())
    
    for i in range(len(original_images)):
        original_image = Image.open(original_images[i].stream)
        styled_image = Image.open(styled_images[i].stream)
        # Create clean filename with counter
        clean_filename = f"processed_{timestamp}_{i}.jpg"
        save_path_original = os.path.join(UPLOAD_DIR_ORIGINAL, clean_filename)
        save_path_style = os.path.join(UPLOAD_DIR_STYLE,clean_filename)
        # Save with explicit format
        original_image.save(save_path_original,format='png')
        styled_image.save(save_path_style,format='png')
        saved_files.append(f'{clean_filename}_original')
        saved_files.append(f'{clean_filename}_styled')
    train_model('ghibli')
    return jsonify({'status': 'success', 'saved_files': saved_files})

if __name__ == '__main__':
    app.run(debug=True)