"""
MedCLIP Image Classification Script

This script provides a command-line interface for medical image classification
using the MedCLIP model with support for custom prompts and flexible configuration.
"""

import torch
import os
import argparse
import json
from typing import Dict, List, Optional, Tuple, Any
from collections import defaultdict
from pathlib import Path

from medclip import MedCLIPModel, MedCLIPVisionModelViT
from medclip import MedCLIPProcessor
from medclip import PromptClassifier
from medclip.prompts import generate_chexpert_class_prompts, process_class_prompts
from PIL import Image


class MedCLIPClassifier:
    """Main class for MedCLIP image classification"""
    
    def __init__(self, device: Optional[str] = None):
        """
        Initialize MedCLIP classifier
        
        Args:
            device: Device to run the model on ('cuda', 'cpu', or None for auto-detect)
        """
        self.device = self._setup_device(device)
        self.processor = None
        self.model = None
        self.classifier = None
        
    def _setup_device(self, device: Optional[str]) -> torch.device:
        """Setup and validate the device for model execution"""
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        device_obj = torch.device(device)
        print(f"Using device: {device_obj}")
        
        if device == 'cuda' and not torch.cuda.is_available():
            print("Warning: CUDA requested but not available, falling back to CPU")
            device_obj = torch.device('cpu')
            
        return device_obj
    
    def load_model(self, ensemble: bool = True) -> None:
        """
        Load the MedCLIP model and classifier
        
        Args:
            ensemble: Whether to use ensemble mode for classification
        """
        print("Initializing MedCLIP model...")
        
        # Initialize processor and model
        self.processor = MedCLIPProcessor()
        self.model = MedCLIPModel(vision_cls=MedCLIPVisionModelViT)
        
        # Load pretrained weights
        print("Loading pretrained weights...")
        self.model.from_pretrained()
        
        # Initialize classifier
        self.classifier = PromptClassifier(self.model, ensemble=ensemble)
        self.classifier = self.classifier.to(self.device)
        
        print("Model loaded successfully!")
    
    def load_image(self, image_path: str) -> Image.Image:
        """
        Load and validate an image
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Loaded PIL Image
            
        Raises:
            FileNotFoundError: If image file doesn't exist
            ValueError: If image cannot be loaded
        """
        # Handle relative paths
        if not os.path.isabs(image_path):
            image_path = os.path.abspath(image_path)
        
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found at {image_path}")
        
        print(f"Loading image from: {image_path}")
        
        try:
            image = Image.open(image_path)
            return image
        except Exception as e:
            raise ValueError(f"Failed to load image: {e}")
    
    def load_custom_prompts(self, prompts_file: str) -> Optional[Dict[str, List[str]]]:
        """
        Load custom prompts from JSON file
        
        Args:
            prompts_file: Path to JSON file containing custom prompts
            
        Returns:
            Dictionary of class names to prompt lists, or None if loading fails
        """
        try:
            with open(prompts_file, 'r', encoding='utf-8') as f:
                prompts_data = json.load(f)
            
            # Validate format
            if not isinstance(prompts_data, dict):
                raise ValueError("Prompts file must contain a dictionary")
            
            for class_name, prompts in prompts_data.items():
                if not isinstance(prompts, list):
                    raise ValueError(f"Prompts for {class_name} must be a list")
                if not all(isinstance(p, str) for p in prompts):
                    raise ValueError(f"All prompts for {class_name} must be strings")
            
            return prompts_data
            
        except Exception as e:
            print(f"Error loading custom prompts: {e}")
            return None
    
    def combine_prompts(self, 
                       custom_prompts: Optional[Dict[str, List[str]]], 
                       default_prompts: Dict[str, List[str]], 
                       num_prompts_per_class: int) -> Dict[str, Any]:
        """
        Combine custom prompts with default CheXpert prompts
        
        Args:
            custom_prompts: Custom prompts from user
            default_prompts: Default CheXpert prompts
            num_prompts_per_class: Target number of prompts per class
            
        Returns:
            Processed prompts ready for model input
        """
        from transformers import AutoTokenizer
        
        tokenizer = AutoTokenizer.from_pretrained('emilyalsentzer/Bio_ClinicalBERT')
        tokenizer.model_max_length = 77
        
        combined_prompts = defaultdict()
        
        # Get all available class names
        all_classes = set(default_prompts.keys())
        if custom_prompts:
            all_classes.update(custom_prompts.keys())
        
        for class_name in all_classes:
            combined_prompt_list = []
            
            # Add default prompts if available
            if class_name in default_prompts:
                default_prompts_for_class = default_prompts[class_name]
                num_default = min(num_prompts_per_class // 2, len(default_prompts_for_class))
                combined_prompt_list.extend(default_prompts_for_class[:num_default])
                print(f"Added {num_default} default prompts for {class_name}")
            
            # Add custom prompts if available
            if custom_prompts and class_name in custom_prompts:
                custom_prompts_for_class = custom_prompts[class_name]
                num_custom = min(num_prompts_per_class - len(combined_prompt_list), 
                               len(custom_prompts_for_class))
                combined_prompt_list.extend(custom_prompts_for_class[:num_custom])
                print(f"Added {num_custom} custom prompts for {class_name}")
            
            # Fill remaining slots with default prompts
            if class_name in default_prompts and len(combined_prompt_list) < num_prompts_per_class:
                remaining_slots = num_prompts_per_class - len(combined_prompt_list)
                default_prompts_for_class = default_prompts[class_name]
                start_idx = len(combined_prompt_list) - (len(combined_prompt_list) - 
                           len(default_prompts_for_class[:num_prompts_per_class // 2]))
                additional_prompts = default_prompts_for_class[start_idx:start_idx + remaining_slots]
                combined_prompt_list.extend(additional_prompts)
                print(f"Added {len(additional_prompts)} additional default prompts for {class_name}")
            
            # Process the combined prompts
            if combined_prompt_list:
                text_inputs = tokenizer(combined_prompt_list, truncation=True, 
                                      padding=True, return_tensors='pt')
                combined_prompts[class_name] = text_inputs
                print(f"Total {len(combined_prompt_list)} prompts for {class_name}")
        
        return combined_prompts
    
    def prepare_inputs(self, 
                      image: Image.Image, 
                      prompts: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """
        Prepare inputs for the model
        
        Args:
            image: Input image
            prompts: Processed prompts
            
        Returns:
            Dictionary of model inputs
        """
        # Process image
        inputs = self.processor(images=image, return_tensors="pt")
        
        # Move inputs to device
        for key in inputs:
            if isinstance(inputs[key], torch.Tensor):
                inputs[key] = inputs[key].to(self.device)
        
        # Add prompts and move to device
        inputs['prompt_inputs'] = prompts
        for key in prompts:
            for tensor_key in prompts[key]:
                if isinstance(prompts[key][tensor_key], torch.Tensor):
                    prompts[key][tensor_key] = prompts[key][tensor_key].to(self.device)
        
        return inputs
    
    def classify(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        """
        Perform classification
        
        Args:
            inputs: Model inputs
            
        Returns:
            Classification results
        """
        print("Running classification...")
        
        with torch.no_grad():
            output = self.classifier(**inputs)
        
        return output
    
    def format_results(self, output: Dict[str, Any], image_path: str) -> Dict[str, Any]:
        """
        Format classification results to match app.py expectations
        
        Args:
            output: Raw model output
            image_path: Path to the input image
            
        Returns:
            Formatted results compatible with app.py
        """
        if 'logits' not in output or 'class_names' not in output:
            return output
        
        logits = output['logits']
        class_names = output['class_names']
        
        # Convert logits to probabilities using softmax
        probabilities = torch.softmax(logits, dim=-1)
        
        # Create confidence scores list
        confidence_scores = []
        for i, (class_name, prob) in enumerate(zip(class_names, probabilities[0])):
            confidence_scores.append({
                "class": class_name,
                "confidence": prob.item()
            })
        
        # Sort by confidence (descending)
        confidence_scores.sort(key=lambda x: x['confidence'], reverse=True)
        
        # Get most likely condition
        most_likely = confidence_scores[0]['class'] if confidence_scores else "Unknown"
        top_confidence = confidence_scores[0]['confidence'] if confidence_scores else 0.0
        
        # Create formatted output
        formatted_results = {
            'image_path': image_path,
            'confidence_scores': confidence_scores,
            'most_likely': most_likely,
            'top_confidence': top_confidence,
            'total_classes': len(class_names),
            'class_names': class_names,
            'logits': logits.cpu().numpy().tolist(),
            'probabilities': probabilities.cpu().numpy().tolist()
        }
        
        # Print detailed results
        print("\n" + "="*50)
        print("CLASSIFICATION RESULTS")
        print("="*50)
        print(f"Most Likely Condition: {most_likely}")
        print(f"Confidence: {top_confidence:.4f}")
        print(f"Total Classes Analyzed: {len(class_names)}")
        print("\nModel Confidence Scores:")
        for i, score in enumerate(confidence_scores, 1):
            print(f"{i}. {score['class']}: {score['confidence']:.4f}")
        print("="*50)
        
        return formatted_results
    
    def save_results(self, results: Dict[str, Any], output_path: str) -> None:
        """
        Save results to JSON file
        
        Args:
            results: Results to save
            output_path: Path to save the results
        """
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            print(f"\nResults saved to: {output_path}")
        except Exception as e:
            print(f"Error saving results: {e}")


class ArgumentParser:
    """Command line argument parser for MedCLIP"""
    
    @staticmethod
    def parse_arguments() -> argparse.Namespace:
        """Parse command line arguments"""
        parser = argparse.ArgumentParser(
            description='MedCLIP Image Classification',
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog="""
Examples:
  python run.py
  python run.py --image path/to/image.jpg
  python run.py --prompts custom_prompts.json --output results.json
  python run.py --image my_image.jpg --prompts my_prompts.json --num_prompts 15
            """
        )
        
        parser.add_argument(
            '--image', 
            type=str, 
            default=None,
            help='Path to the input image (if not provided, uses default example image)'
        )
        
        parser.add_argument(
            '--prompts', 
            type=str, 
            default=None,
            help='Path to JSON file containing custom prompts'
        )
        
        parser.add_argument(
            '--num_prompts', 
            type=int, 
            default=10,
            help='Number of prompts per class (default: 10)'
        )
        
        parser.add_argument(
            '--ensemble', 
            action='store_true',
            default=True,
            help='Use ensemble mode for classification (default: True)'
        )
        
        parser.add_argument(
            '--output', 
            type=str, 
            default=None,
            help='Path to save results as JSON file (optional)'
        )
        
        parser.add_argument(
            '--no-save',
            action='store_true',
            help='Do not save results to file (useful for API calls)'
        )
        
        parser.add_argument(
            '--device',
            type=str,
            default=None,
            choices=['cuda', 'cpu'],
            help='Device to run the model on (auto-detect if not specified)'
        )
        
        return parser.parse_args()


def get_default_image_path() -> str:
    """Get the default example image path"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(script_dir, 'example_data', 'view1_frontal.jpg')


def main():
    """Main execution function"""
    # Parse arguments
    args = ArgumentParser.parse_arguments()
    
    try:
        # Initialize classifier
        classifier = MedCLIPClassifier(device=args.device)
        
        # Load model
        classifier.load_model(ensemble=args.ensemble)
        
        # Load image
        image_path = args.image if args.image else get_default_image_path()
        image = classifier.load_image(image_path)
        
        # Prepare prompts
        print("Generating default CheXpert class prompts...")
        default_prompts = generate_chexpert_class_prompts(n=args.num_prompts)
        
        if args.prompts:
            print("Loading custom prompts...")
            # Handle both file path and direct prompt list
            if isinstance(args.prompts, str) and os.path.exists(args.prompts):
                # Load from file
                custom_prompts = classifier.load_custom_prompts(args.prompts)
            else:
                # Handle direct prompt list (for app.py integration)
                try:
                    if isinstance(args.prompts, list):
                        # Convert list of prompts to custom format
                        custom_prompts = {"Custom": args.prompts}
                    else:
                        # Try to parse as JSON string
                        import json
                        prompt_data = json.loads(args.prompts)
                        custom_prompts = prompt_data if isinstance(prompt_data, dict) else {"Custom": prompt_data}
                except:
                    print("Failed to parse custom prompts, using only default prompts...")
                    custom_prompts = None
            
            if custom_prompts:
                print("Combining custom prompts with default CheXpert prompts...")
                prompts = classifier.combine_prompts(custom_prompts, default_prompts, args.num_prompts)
                print(f"Combined prompts for {len(prompts)} classes")
            else:
                print("Failed to load custom prompts, using only default CheXpert prompts...")
                prompts = process_class_prompts(default_prompts)
        else:
            print("Using only default CheXpert prompts...")
            prompts = process_class_prompts(default_prompts)
        
        # Prepare inputs
        inputs = classifier.prepare_inputs(image, prompts)
        
        # Perform classification
        output = classifier.classify(inputs)
        
        # Format and display results
        results = classifier.format_results(output, image_path)
        
        # Save results if requested
        if args.output and not args.no_save:
            classifier.save_results(results, args.output)
        
        # Return results in the format expected by app.py
        return {
            "status": "completed",
            "data": results,
            "raw_output": "Classification completed successfully"
        }
        
    except Exception as e:
        print(f"Error occurred: {str(e)}")
        import traceback
        traceback.print_exc()
        return {
            "status": "failed",
            "error": str(e),
            "data": {}
        }


if __name__ == "__main__":
    main()