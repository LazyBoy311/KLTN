"""
Adapter for open_clip models to work with IBA functions
"""

import torch
import torch.nn as nn
import copy
from typing import Optional, Tuple

class OpenCLIPAdapter(nn.Module):
    """
    Adapter to make open_clip models compatible with IBA functions
    """
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.vision_model = self._wrap_vision_model(model.visual)
        self.text_model = self._wrap_text_model(model)
        
    def _wrap_vision_model(self, vision_model):
        """Wrap vision model to match transformers interface"""
        class VisionWrapper(nn.Module):
            def __init__(self, model):
                super().__init__()
                self.model = model
                
            def forward(self, x, output_hidden_states=False, **kwargs):
                if output_hidden_states:
                    # For open_clip, we need to extract intermediate features
                    # This is a simplified version - may need adjustment based on specific model
                    features = []
                    x = self.model.patch_embed(x)
                    x = self.model.pos_drop(x)
                    
                    for i, block in enumerate(self.model.blocks):
                        x = block(x)
                        if i >= len(self.model.blocks) - 4:  # Last 4 layers
                            features.append(x)
                    
                    # Add final output
                    features.append(x)
                    
                    return type('obj', (object,), {
                        'hidden_states': features,
                        'last_hidden_state': x
                    })()
                else:
                    return self.model(x)
                    
        return VisionWrapper(vision_model)
    
    def _wrap_text_model(self, model):
        """Wrap text model to match transformers interface"""
        class TextWrapper(nn.Module):
            def __init__(self, model):
                super().__init__()
                self.model = model
                
            def forward(self, x, output_hidden_states=False, **kwargs):
                if output_hidden_states:
                    # For open_clip text model
                    features = []
                    x = self.model.token_embedding(x)
                    x = self.model.positional_embedding(x)
                    
                    for i, block in enumerate(self.model.transformer.resblocks):
                        x = block(x)
                        if i >= len(self.model.transformer.resblocks) - 4:  # Last 4 layers
                            features.append(x)
                    
                    # Add final output
                    features.append(x)
                    
                    return type('obj', (object,), {
                        'hidden_states': features,
                        'last_hidden_state': x
                    })()
                else:
                    return self.model(x)
                    
        return TextWrapper(model.text)
    
    def get_image_features(self, x, **kwargs):
        """Get image features"""
        return self.model.encode_image(x, normalize=False)
    
    def get_text_features(self, x, **kwargs):
        """Get text features"""
        return self.model.encode_text(x, normalize=False)
    
    def forward(self, input_ids=None, pixel_values=None, **kwargs):
        """Forward pass compatible with transformers interface"""
        if pixel_values is not None:
            image_features = self.get_image_features(pixel_values)
        else:
            image_features = None
            
        if input_ids is not None:
            text_features = self.get_text_features(input_ids)
        else:
            text_features = None
            
        return type('obj', (object,), {
            'image_embeds': image_features,
            'text_embeds': text_features
        })()

def create_openclip_adapter(model):
    """Create adapter for open_clip model"""
    return OpenCLIPAdapter(model) 