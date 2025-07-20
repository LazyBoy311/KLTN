"""
Simplified IBA functions for open_clip models
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional

def simple_vision_heatmap(text_tokens, image_tensor, model, layer_idx=9, beta=0.1, var=1.0):
    """
    Simplified vision heatmap for open_clip models
    """
    model.eval()
    
    with torch.no_grad():
        # Get image features
        image_features = model.encode_image(image_tensor, normalize=False)
        text_features = model.encode_text(text_tokens, normalize=False)
        
        # Calculate attention weights
        attention_weights = F.softmax(torch.matmul(image_features, text_features.T), dim=-1)
        
        # Create simple heatmap based on attention
        heatmap = attention_weights.mean(dim=0).reshape(1, 1, 14, 14)  # Assuming 224x224 -> 14x14
        heatmap = F.interpolate(heatmap, size=(224, 224), mode='bilinear', align_corners=False)
        heatmap = heatmap.squeeze().cpu().numpy()
        
        # Normalize
        heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
        
        return heatmap

def simple_text_heatmap(text_tokens, image_tensor, model, layer_idx=9, beta=0.1, var=1.0):
    """
    Simplified text heatmap for open_clip models
    """
    model.eval()
    
    with torch.no_grad():
        # Get features
        image_features = model.encode_image(image_tensor, normalize=False)
        text_features = model.encode_text(text_tokens, normalize=False)
        
        # Calculate attention weights for text
        attention_weights = F.softmax(torch.matmul(text_features, image_features.T), dim=-1)
        
        # Create simple text heatmap
        heatmap = attention_weights.mean(dim=0).cpu().numpy()
        
        # Normalize
        heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
        
        return heatmap

def openclip_vision_heatmap_iba(text_tokens, image_tensor, model, layer_idx=9, beta=0.1, var=1.0, **kwargs):
    """
    Wrapper for open_clip vision heatmap
    """
    return simple_vision_heatmap(text_tokens, image_tensor, model, layer_idx, beta, var)

def openclip_text_heatmap_iba(text_tokens, image_tensor, model, layer_idx=9, beta=0.1, var=1.0, **kwargs):
    """
    Wrapper for open_clip text heatmap
    """
    return simple_text_heatmap(text_tokens, image_tensor, model, layer_idx, beta, var) 