import torch
import torch.nn as nn
import torch.optim as optim
from collections import defaultdict
import numpy as np
import wandb
import os
from torchvision.utils import save_image
import torch.nn.functional as F

def sample_recon(model, data, accelerator, output_path, num_samples=4):
    model.eval()
    with torch.no_grad():
        current_frames, reference_frames = data
        batch_size = current_frames.size(0)
        num_samples = min(num_samples, batch_size)
        current_frames, reference_frames = current_frames[:num_samples], reference_frames[:num_samples]
        
        # Get reconstructed frames from IMF
        reconstructed_frames = model(current_frames, reference_frames)
        
        # Ensure reconstructed_frames have the same size as reference_frames
        if reconstructed_frames.shape != current_frames.shape:
            reconstructed_frames = F.interpolate(reconstructed_frames, size=current_frames.shape[2:], mode='bilinear', align_corners=False)
        
        # Reduce size by half
        # current_frames = F.interpolate(current_frames, scale_factor=0.5, mode='bilinear', align_corners=False)
        # reference_frames = F.interpolate(reference_frames, scale_factor=0.5, mode='bilinear', align_corners=False)
        # reconstructed_frames = F.interpolate(reconstructed_frames, scale_factor=0.5, mode='bilinear', align_corners=False)
        
        # Unnormalize reconstructed frames
        # reconstructed_frames = reconstructed_frames * 0.5 + 0.5
        # reconstructed_frames = torch.clamp(reconstructed_frames, 0, 1)
        

        
        # Prepare frames for saving (2x4 grid)
        frames = torch.cat((reconstructed_frames, reference_frames), dim=0)
        
        # Ensure we have a valid output directory
        if output_path:
            output_dir = os.path.dirname(output_path)
            if not output_dir:
                output_dir = '.'
            os.makedirs(output_dir, exist_ok=True)
            
            # Save frames as a grid (2x4)
            save_image(accelerator.gather(frames), output_path, nrow=4, padding=2, normalize=False)
            # accelerator.print(f"Saved sample reconstructions to {output_path}")
        else:
            accelerator.print("Warning: No output path provided. Skipping image save.")
        
        # Log images to wandb
        wandb_images = []
        for i in range(num_samples):
            wandb_images.extend([
                wandb.Image(reconstructed_frames[i].cpu().detach().numpy().transpose(1, 2, 0), caption=f"Reconstructed {i}"),
                wandb.Image(reference_frames[i].cpu().detach().numpy().transpose(1, 2, 0), caption=f"Reference {i}")
            ])
        
        wandb.log({"Sample Reconstructions": wandb_images})
        
        return frames


def monitor_gradients(model, epoch, batch_idx, log_interval=10):
    """
    Monitor gradients of the model parameters.
    
    :param model: The neural network model
    :param epoch: Current epoch number
    :param batch_idx: Current batch index
    :param log_interval: How often to log gradient statistics
    """
    if batch_idx % log_interval == 0:
        grad_stats = defaultdict(list)
        
        for name, param in model.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.norm().item()
                grad_stats['norm'].append(grad_norm)
                
                if torch.isnan(param.grad).any():
                    print(f"NaN gradient detected in {name}")
                
                if torch.isinf(param.grad).any():
                    print(f"Inf gradient detected in {name}")
                
                grad_stats['names'].append(name)
        
        if grad_stats['norm']:
            avg_norm = np.mean(grad_stats['norm'])
            max_norm = np.max(grad_stats['norm'])
            min_norm = np.min(grad_stats['norm'])
            
            print(f"Epoch {epoch}, Batch {batch_idx}")
            print(f"Gradient norms - Avg: {avg_norm:.4f}, Max: {max_norm:.4f}, Min: {min_norm:.4f}")
            
            # Identify layers with unusually high or low gradients
            threshold_high = avg_norm * 10  # Adjust this multiplier as needed
            threshold_low = avg_norm * 0.1  # Adjust this multiplier as needed
            
            for name, norm in zip(grad_stats['names'], grad_stats['norm']):
                if norm > threshold_high:
                    print(f"High gradient in {name}: {norm:.4f}")
                elif norm < threshold_low:
                    print(f"Low gradient in {name}: {norm:.4f}")
        else:
            print("No gradients to monitor")



# helper for gradient vanishing / explosion
def hook_fn(name):
    def hook(grad):
        if torch.isnan(grad).any():
            # print(f"🔥 NaN gradient detected in {name}")
            return torch.zeros_like(grad)  # Replace NaN with zero
        elif torch.isinf(grad).any():
            # print(f"🔥 Inf gradient detected in {name}")
            return torch.clamp(grad, -1e6, 1e6)  # Clamp infinite values
        #else:
            # You can add more conditions or logging here
         #  grad_norm = grad.norm().item()
         #   print(f"Gradient norm for {name}: {grad_norm}")
        return grad
    return hook

def add_gradient_hooks(model):
    for name, param in model.named_parameters():
        if param.requires_grad:
            param.register_hook(hook_fn(name))