"""
MedCLIP-SAMv2: Medical Image Segmentation using CLIP and SAM

This script provides a command-line interface for medical image segmentation using
BiomedCLIP for saliency mapping and SAM for precise segmentation.
Supports custom text prompts and flexible configuration.
"""

import os
import sys
import warnings
import argparse
import json
from typing import Optional, Tuple, List, Dict, Any

import torch
import torchvision
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image, ImageOps
from sklearn.cluster import KMeans

# Transformers imports
from transformers import (
    CLIPProcessor,
    AutoModel,
    AutoProcessor,
    AutoTokenizer,
    CLIPModel
)

# Local imports - fix paths
sys.path.append(os.path.join(os.path.dirname(__file__), 'saliency_maps'))
from scripts.plot import (
    generate_shades_with_alpha,
    plot_text_with_colors,
    visualize_vandt_heatmap
)
from scripts.methods import vision_heatmap_iba, text_heatmap_iba

# SAM imports
from segment_anything import sam_model_registry, SamPredictor

# Suppress warnings
warnings.filterwarnings('ignore')
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Global variables
_CONTOUR_INDEX = 1 if cv2.__version__.split('.')[0] == '3' else 0


class MedCLIPSAMv2Pipeline:
    """Main class for MedCLIP-SAMv2 image segmentation"""
    
    def __init__(self, device: Optional[str] = None):
        """
        Initialize MedCLIP-SAMv2 pipeline
        
        Args:
            device: Device to run the model on ('cuda', 'cpu', or None for auto-detect)
        """
        self.device = self._setup_device(device)
        self.model = None
        self.processor = None
        self.processor2 = None
        self.tokenizer = None
        
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
    
    def init_model(self):
        """Initialize BiomedCLIP model and processors."""
        print("Initializing BiomedCLIP model...")

        self.model = AutoModel.from_pretrained(
            "chuhac/BiomedCLIP-vit-bert-hf",
            trust_remote_code=True
        ).to(self.device)
        
        self.processor = AutoProcessor.from_pretrained(
            "chuhac/BiomedCLIP-vit-bert-hf",
            trust_remote_code=True
        )
        
        self.processor2 = CLIPProcessor.from_pretrained(
            "openai/clip-vit-base-patch16"
        )
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            "chuhac/BiomedCLIP-vit-bert-hf",
            trust_remote_code=True
        )
        
        print("BiomedCLIP model initialized successfully!")
        return self.model, self.processor, self.processor2, self.tokenizer
    
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
    
    def plot(
        self,
        image_path: str,
        text: str,
        vbeta: float = 0.1,
        vvar: float = 1.0,
        vlayer: int = 9,
        tbeta: float = 0.1,
        tvar: float = 1.0,
        tlayer: int = 9
    ) -> np.ndarray:
        """
        Generate saliency map using BiomedCLIP.
        
        Args:
            image_path: Path to input image
            text: Text description for the image
            vbeta, vvar, vlayer: Vision heatmap parameters
            tbeta, tvar, tlayer: Text heatmap parameters
        
        Returns:
            Saliency map as numpy array
        """
        if self.model is None:
            self.init_model()
        
        # Preprocess image
        image = Image.open(image_path).convert('RGB')
        image_feat = self.processor(
            images=image,
            return_tensors="pt"
        )['pixel_values'].to(self.device)
        
        # Tokenize text
        text_ids = torch.tensor([
            self.tokenizer.encode(text, add_special_tokens=True)
        ]).to(self.device)
        text_words = self.tokenizer.convert_ids_to_tokens(text_ids[0].tolist())
        
        # Train information bottleneck on image
        print("Training M2IB on the image...")
        vmap = vision_heatmap_iba(text_ids, image_feat, self.model, vlayer, vbeta, vvar)
        
        # Train information bottleneck on text
        print("Training M2IB on the text...")
        tmap = text_heatmap_iba(text_ids, image_feat, self.model, tlayer, tbeta, tvar)
        
        # Visualize heatmaps
        image_under = self.processor2(
            images=image,
            return_tensors="pt",
            do_normalize=False
        )['pixel_values'][0].permute(1, 2, 0)
        visualize_vandt_heatmap(tmap, vmap, text_words, image_under)
        
        # Resize saliency map to original image size
        img = np.array(image)
        vmap = cv2.resize(
            np.array(vmap),
            (img.shape[1], img.shape[0]),
            interpolation=cv2.INTER_NEAREST
        ) * 255
        
        return vmap
    
    def postprocess_saliency_map(self, vmap: np.ndarray, output_dir: str = ".") -> np.ndarray:
        """
        Post-process saliency map using K-means clustering.
        
        Args:
            vmap: Input saliency map
            output_dir: Directory to save the postprocessed map
        
        Returns:
            Post-processed segmentation mask
        """
        kmeans = KMeans(n_clusters=2, random_state=10)
        attn_weights = vmap / 255
        
        # Keep only high attention weight scores
        h, w = attn_weights.shape
        filtered_attn_weights = attn_weights > 0.4
        attn_weights = attn_weights * filtered_attn_weights
        
        # Resize for clustering
        image = cv2.resize(
            attn_weights,
            (256, 256),
            interpolation=cv2.INTER_NEAREST
        )
        flat_image = image.reshape(-1, 1)
        
        # Apply K-means clustering
        labels = kmeans.fit_predict(flat_image)
        segmented_image = labels.reshape(256, 256)
        
        # Identify background cluster
        centroids = kmeans.cluster_centers_.flatten()
        background_cluster = np.argmin(centroids)
        
        # Mark background pixels as 0 and foreground pixels as 1
        segmented_image = np.where(segmented_image == background_cluster, 0, 1)
        
        # Resize back to original size
        segmented_image = cv2.resize(
            segmented_image,
            (w, h),
            interpolation=cv2.INTER_NEAREST
        )
        segmented_image = segmented_image.astype(np.uint8) * 255
        
        # Connected components analysis
        nb_blobs, im_with_separated_blobs, stats, _ = cv2.connectedComponentsWithStats(
            segmented_image
        )
        sizes = stats[:, cv2.CC_STAT_AREA]
        
        # Sort sizes (ignoring the background at index 0)
        sorted_sizes = sorted(sizes[1:], reverse=True)
        
        # Keep only the largest contour
        num_contours = 1
        top_k_sizes = sorted_sizes[:num_contours]
        
        im_result = np.zeros_like(im_with_separated_blobs)
        
        for index_blob in range(1, nb_blobs):
            if sizes[index_blob] in top_k_sizes:
                im_result[im_with_separated_blobs == index_blob] = 255
        
        segmented_image = im_result
        
        # Save postprocessed map
        output_path = os.path.join(output_dir, "postprocessed_map.png")
        cv2.imwrite(output_path, segmented_image)
        
        plt.imshow(segmented_image, cmap="gray")
        return segmented_image
    
    def scoremap2bbox(
        self,
        scoremap: np.ndarray,
        multi_contour_eval: bool = False
    ) -> Tuple[List[List[int]], List, int]:
        """
        Convert scoremap to bounding boxes.
        
        Args:
            scoremap: Input scoremap
            multi_contour_eval: Whether to evaluate multiple contours
        
        Returns:
            Tuple of (bounding_boxes, contours, num_contours)
        """
        height, width = scoremap.shape
        scoremap_image = (scoremap * 255).astype(np.uint8)
        
        contours, _ = cv2.findContours(
            image=scoremap_image,
            mode=cv2.RETR_EXTERNAL,
            method=cv2.CHAIN_APPROX_SIMPLE
        )
        
        num_contours = len(contours)
        
        if len(contours) == 0:
            return np.asarray([[0, 0, width, height]]), [], 1
        
        if not multi_contour_eval:
            contours = [np.concatenate(contours)]
        
        estimated_boxes = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            x0, y0, x1, y1 = x, y, x + w, y + h
            x1 = min(x1, width - 1)
            y1 = min(y1, height - 1)
            estimated_boxes.append([x0, y0, x1, y1])
        
        return estimated_boxes, contours, num_contours
    
    def segment_image(
        self,
        image_path: str,
        output_dir: str,
        custom_text: Optional[str] = None,
        enable_logging: bool = True
    ) -> np.ndarray:
        """
        Segment image using SAM with saliency map guidance.
        
        Args:
            image_path: Path to input image
            output_dir: Directory to save outputs
            custom_text: Custom text description
            enable_logging: Enable logging
        
        Returns:
            Segmentation mask
        """
        # SAM configuration
        sam_checkpoint = os.path.join(
            os.path.dirname(__file__),
            "segment-anything",
            "sam_vit_h_4b8939.pth"
        )
        
        # Check if SAM checkpoint exists
        if not os.path.exists(sam_checkpoint):
            raise FileNotFoundError(f"SAM checkpoint not found: {sam_checkpoint}")
        
        print(f"Loading SAM checkpoint from: {sam_checkpoint}")
        
        model_type = "vit_h"
        device = "cuda"
        
        # Parameters
        prompts = "boxes"  # ["boxes", "points", "both"]
        pos_num_points = 10
        neg_num_points = 2
        
        # Initialize SAM
        try:
            print(f"Attempting to load SAM model from: {sam_checkpoint}")
            sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
            sam.to(device=device)
            predictor = SamPredictor(sam)
            print("SAM model loaded successfully!")
        except Exception as e:
            print(f"Error loading SAM model: {e}")
            print(f"Check if the file exists and is accessible: {sam_checkpoint}")
            raise
        
        # Load image and mask
        img = cv2.imread(image_path)
        mask_path = os.path.join(output_dir, "postprocessed_map.png")
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        predictor.set_image(img)
        
        # Get bounding boxes from saliency map
        bounding_boxes, contours, num_contours = self.scoremap2bbox(
            mask,
            multi_contour_eval=False
        )
        
        if prompts == "boxes":
            bounding_boxes = np.array(bounding_boxes)
            input_boxes = torch.tensor(bounding_boxes, device=device)
            transformed_boxes = predictor.transform.apply_boxes_torch(
                input_boxes,
                img.shape[:2]
            )
            masks, scores, _ = predictor.predict_torch(
                point_coords=None,
                point_labels=None,
                boxes=transformed_boxes,
                multimask_output=False
            )
            masks = masks.cpu().numpy()
            
        elif prompts == "points":
            # Generate positive and negative points
            pos_random_points = []
            neg_random_points = []
            
            candidate_points = np.argwhere(mask.transpose(1, 0) > 0)
            h, w = mask.shape
            random_index = np.random.choice(
                len(candidate_points),
                pos_num_points,
                replace=False
            )
            pos_random_points = candidate_points[random_index]
            
            candidate_points = np.argwhere(mask.transpose(1, 0) == 0)
            random_index = np.random.choice(
                len(candidate_points),
                neg_num_points,
                replace=False
            )
            neg_random_points = candidate_points[random_index]
            
            all_random_points = np.concatenate([pos_random_points, neg_random_points])
            all_input_labels = [1] * len(pos_random_points) + [0] * len(neg_random_points)
            
            masks, scores, _ = predictor.predict(
                point_coords=all_random_points,
                point_labels=all_input_labels,
                multimask_output=False,
            )
            
        else:  # "both"
            input_boxes = torch.tensor(bounding_boxes, device=device)
            
            # Generate points
            pos_random_points = []
            neg_random_points = []
            
            candidate_points = np.argwhere(mask.transpose(1, 0) > 0)
            h, w = mask.shape
            random_index = np.random.choice(
                len(candidate_points),
                pos_num_points,
                replace=False
            )
            pos_random_points = candidate_points[random_index]
            
            candidate_points = np.argwhere(mask.transpose(1, 0) == 0)
            random_index = np.random.choice(
                len(candidate_points),
                neg_num_points,
                replace=False
            )
            neg_random_points = candidate_points[random_index]
            
            all_random_points = np.concatenate([pos_random_points, neg_random_points])
            all_input_labels = [1] * len(pos_random_points) + [0] * len(neg_random_points)
            
            input_points = torch.tensor(all_random_points, device=device)
            input_labels = torch.tensor(all_input_labels, device=device)
            input_points = input_points.repeat((len(bounding_boxes), 1, 1))
            input_labels = input_labels.repeat((len(bounding_boxes), 1))
            
            transformed_boxes = predictor.transform.apply_boxes_torch(
                input_boxes,
                img.shape[:2]
            )
            transformed_points = predictor.transform.apply_coords_torch(
                input_points,
                img.shape[:2]
            )
            
            masks, scores, _ = predictor.predict_torch(
                point_coords=transformed_points,
                point_labels=input_labels,
                boxes=transformed_boxes,
                multimask_output=False
            )
            masks = masks.cpu().numpy()
        
        # Save output
        output_path = os.path.join(output_dir, "sam_output.png")
        mask_sam = np.squeeze(masks * 255).astype('uint8')
        cv2.imwrite(output_path, mask_sam)
        plt.imshow(mask_sam, cmap="gray")
        
        return mask_sam
    
    def run_pipeline(
        self,
        image_path: str,
        output_dir: str,
        custom_text: Optional[str] = None,
        enable_logging: bool = True
    ) -> Dict[str, Any]:
        """
        Run complete MedCLIP-SAMv2 pipeline.
        
        Args:
            image_path: Path to input image
            output_dir: Directory to save outputs
            custom_text: Custom text description
            enable_logging: Enable logging
        
        Returns:
            Dictionary containing pipeline results
        """
        try:
            # Create output directory if it doesn't exist
            os.makedirs(output_dir, exist_ok=True)
            
            # Check if image file exists
            if not os.path.exists(image_path):
                raise FileNotFoundError(f"Image file not found: {image_path}")
            
            print(f"Processing image: {image_path}")
            print(f"Output directory: {output_dir}")
            
            # Initialize model if needed
            if self.model is None:
                self.init_model()
            
            # Use default text if none provided
            if custom_text is None:
                custom_text = "A medical brain MRI scan showing a well-circumscribed, extra-axial mass suggestive of a meningioma tumor."
            
            # Step 1: Generate saliency map
            print("Step 1: Generating saliency map...")
            vmap = self.plot(image_path, custom_text)
            
            # Step 2: Post-process saliency map
            print("Step 2: Post-processing saliency map...")
            segmented_image = self.postprocess_saliency_map(vmap, output_dir)
            
            # Step 3: Get bounding boxes
            print("Step 3: Extracting bounding boxes...")
            bounding_boxes, contours, num_contours = self.scoremap2bbox(
                segmented_image,
                multi_contour_eval=False
            )
            
            # Step 4: Run SAM segmentation
            print("Step 4: Running SAM segmentation...")
            segmentation_result = self.segment_image(image_path, output_dir, custom_text, enable_logging)
            
            # Check for output files
            output_files = {}
            sam_output_path = os.path.join(output_dir, "sam_output.png")
            if os.path.exists(sam_output_path):
                output_files["sam_output"] = sam_output_path
            
            postprocessed_path = os.path.join(output_dir, "postprocessed_map.png")
            if os.path.exists(postprocessed_path):
                output_files["postprocessed_map"] = postprocessed_path
            
            print(f"Pipeline completed successfully!")
            print(f"Number of contours found: {num_contours}")
            print(f"Output files: {list(output_files.keys())}")
            
            return {
                "status": "completed",
                "image_path": image_path,
                "output_dir": output_dir,
                "custom_text": custom_text,
                "num_contours": num_contours,
                "bounding_boxes": bounding_boxes,
                "output_files": output_files,
                "segmentation_result": segmentation_result
            }
            
        except Exception as e:
            print(f"Pipeline failed: {e}")
            return {
                "status": "failed",
                "error": str(e),
                "image_path": image_path,
                "output_dir": output_dir
            }


class ArgumentParser:
    """Command line argument parser for MedCLIP-SAMv2"""
    
    @staticmethod
    def parse_arguments() -> argparse.Namespace:
        """Parse command line arguments"""
        parser = argparse.ArgumentParser(
            description='MedCLIP-SAMv2 Image Segmentation',
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog="""
Examples:
  python run.py
  python run.py --image path/to/image.jpg
  python run.py --image my_image.jpg --text "tumor in brain MRI"
  python run.py --image my_image.jpg --output results/
            """
        )
        
        parser.add_argument(
            '--image', 
            type=str, 
            default=None,
            help='Path to the input image (if not provided, uses default example image)'
        )
        
        parser.add_argument(
            '--text', 
            type=str, 
            default=None,
            help='Custom text description for the image'
        )
        
        parser.add_argument(
            '--output', 
            type=str, 
            default=None,
            help='Output directory for results (default: current directory)'
        )
        
        parser.add_argument(
            '--device',
            type=str,
            default=None,
            choices=['cuda', 'cpu'],
            help='Device to run the model on (auto-detect if not specified)'
        )
        
        parser.add_argument(
            '--no-logging',
            action='store_true',
            help='Disable logging output'
        )
        
        return parser.parse_args()


def get_default_image_path() -> str:
    """Get the default example image path"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(script_dir, 'assets', 'example.png')


def main():
    """Main execution function"""
    # Parse arguments
    args = ArgumentParser.parse_arguments()
    
    try:
        # Initialize pipeline
        pipeline = MedCLIPSAMv2Pipeline(device=args.device)
        
        # Set image path
        image_path = args.image if args.image else get_default_image_path()
        
        # Set output directory
        output_dir = args.output if args.output else os.path.dirname(__file__)
        
        # Set custom text
        custom_text = args.text
        
        # Set logging
        enable_logging = not args.no_logging
        
        # Run pipeline
        results = pipeline.run_pipeline(
            image_path=image_path,
            output_dir=output_dir,
            custom_text=custom_text,
            enable_logging=enable_logging
        )
        
        # Print results
        if results["status"] == "completed":
            print("\n" + "="*50)
            print("SEGMENTATION RESULTS")
            print("="*50)
            print(f"Status: {results['status']}")
            print(f"Image: {results['image_path']}")
            print(f"Output Directory: {results['output_dir']}")
            print(f"Number of Contours: {results['num_contours']}")
            print(f"Output Files: {list(results['output_files'].keys())}")
            print("="*50)
        else:
            print(f"\nPipeline failed: {results.get('error', 'Unknown error')}")
        
        return results
        
    except Exception as e:
        print(f"Error occurred: {str(e)}")
        import traceback
        traceback.print_exc()
        return {
            "status": "failed",
            "error": str(e)
        }

# Cách chạy script này:
# python run.py --image <đường_dẫn_ảnh> --output <thư_mục_output> --text "<mô_tả_tùy_chỉnh>" [--no_logging]
# Ví dụ:
# python run.py --image ./data/test.png --output ./results --text "A medical image showing a tumor"

if __name__ == "__main__":
    try:
        result = main()
        print("Pipeline completed successfully!")
    except Exception as e:
        print(f"Error running pipeline: {e}")
        import traceback
        traceback.print_exc()

