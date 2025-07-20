"""
MedCLIP-SAMv2 Batch Processing Script

This script allows processing multiple images with pre-prepared prompts
while loading the model only once for efficiency.
"""

import os
import sys
import json
import argparse
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging

# Add the current directory to path for imports
sys.path.append(os.path.dirname(__file__))
from run import MedCLIPSAMv2Pipeline

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('batch_processing.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class BatchProcessor:
    """Batch processor for MedCLIP-SAMv2"""
    
    def __init__(self, device: Optional[str] = None, max_workers: int = 1):
        """
        Initialize batch processor
        
        Args:
            device: Device to run models on
            max_workers: Maximum number of parallel workers (1 for GPU, more for CPU)
        """
        self.device = device
        self.max_workers = max_workers
        self.pipeline = None
        self.results = []
        
    def initialize_pipeline(self):
        """Initialize the MedCLIP-SAMv2 pipeline once"""
        logger.info("Initializing MedCLIP-SAMv2 pipeline...")
        self.pipeline = MedCLIPSAMv2Pipeline(device=self.device)
        self.pipeline.init_model()
        logger.info("Pipeline initialized successfully!")
        
    def process_single_image(
        self, 
        image_path: str, 
        text_prompt: str, 
        output_dir: str
    ) -> Dict[str, Any]:
        """
        Process a single image with given text prompt
        
        Args:
            image_path: Path to input image
            text_prompt: Text description for the image
            output_dir: Output directory for results
            
        Returns:
            Dictionary containing processing results
        """
        try:
            logger.info(f"Processing image: {image_path}")
            start_time = time.time()
            
            # Create image-specific output directory
            image_name = Path(image_path).stem
            image_output_dir = os.path.join(output_dir, image_name)
            os.makedirs(image_output_dir, exist_ok=True)
            
            # Run pipeline
            result = self.pipeline.run_pipeline(
                image_path=image_path,
                output_dir=image_output_dir,
                custom_text=text_prompt,
                enable_logging=False  # Reduce logging noise in batch mode
            )
            
            processing_time = time.time() - start_time
            result['processing_time'] = processing_time
            result['image_name'] = image_name
            
            if result['status'] == 'completed':
                logger.info(f"✓ Completed {image_name} in {processing_time:.2f}s")
            else:
                logger.error(f"✗ Failed {image_name}: {result.get('error', 'Unknown error')}")
                
            return result
            
        except Exception as e:
            logger.error(f"Error processing {image_path}: {str(e)}")
            return {
                'status': 'failed',
                'error': str(e),
                'image_path': image_path,
                'text_prompt': text_prompt,
                'processing_time': 0,
                'image_name': Path(image_path).stem
            }
    
    def process_batch(
        self, 
        image_prompt_pairs: List[Tuple[str, str]], 
        output_dir: str,
        use_parallel: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Process a batch of images with their corresponding prompts
        
        Args:
            image_prompt_pairs: List of (image_path, text_prompt) tuples
            output_dir: Base output directory
            use_parallel: Whether to use parallel processing
            
        Returns:
            List of processing results
        """
        if self.pipeline is None:
            self.initialize_pipeline()
        
        logger.info(f"Starting batch processing of {len(image_prompt_pairs)} images")
        logger.info(f"Output directory: {output_dir}")
        logger.info(f"Parallel processing: {use_parallel}")
        
        os.makedirs(output_dir, exist_ok=True)
        
        if use_parallel and self.max_workers > 1:
            return self._process_parallel(image_prompt_pairs, output_dir)
        else:
            return self._process_sequential(image_prompt_pairs, output_dir)
    
    def _process_sequential(
        self, 
        image_prompt_pairs: List[Tuple[str, str]], 
        output_dir: str
    ) -> List[Dict[str, Any]]:
        """Process images sequentially"""
        results = []
        total_start_time = time.time()
        
        for i, (image_path, text_prompt) in enumerate(image_prompt_pairs, 1):
            logger.info(f"Processing {i}/{len(image_prompt_pairs)}: {Path(image_path).name}")
            
            result = self.process_single_image(image_path, text_prompt, output_dir)
            results.append(result)
            
            # Progress update
            if i % 10 == 0 or i == len(image_prompt_pairs):
                elapsed = time.time() - total_start_time
                avg_time = elapsed / i
                remaining = avg_time * (len(image_prompt_pairs) - i)
                logger.info(f"Progress: {i}/{len(image_prompt_pairs)} "
                          f"(Avg: {avg_time:.2f}s, ETA: {remaining:.1f}s)")
        
        total_time = time.time() - total_start_time
        logger.info(f"Batch processing completed in {total_time:.2f}s")
        
        return results
    
    def _process_parallel(
        self, 
        image_prompt_pairs: List[Tuple[str, str]], 
        output_dir: str
    ) -> List[Dict[str, Any]]:
        """Process images in parallel (use with caution for GPU)"""
        results = []
        total_start_time = time.time()
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all tasks
            future_to_pair = {
                executor.submit(
                    self.process_single_image, 
                    image_path, 
                    text_prompt, 
                    output_dir
                ): (image_path, text_prompt) 
                for image_path, text_prompt in image_prompt_pairs
            }
            
            # Collect results as they complete
            for i, future in enumerate(as_completed(future_to_pair), 1):
                image_path, text_prompt = future_to_pair[future]
                try:
                    result = future.result()
                    results.append(result)
                    
                    if i % 10 == 0 or i == len(image_prompt_pairs):
                        elapsed = time.time() - total_start_time
                        avg_time = elapsed / i
                        remaining = avg_time * (len(image_prompt_pairs) - i)
                        logger.info(f"Progress: {i}/{len(image_prompt_pairs)} "
                                  f"(Avg: {avg_time:.2f}s, ETA: {remaining:.1f}s)")
                        
                except Exception as e:
                    logger.error(f"Error in parallel processing: {e}")
                    results.append({
                        'status': 'failed',
                        'error': str(e),
                        'image_path': image_path,
                        'text_prompt': text_prompt
                    })
        
        total_time = time.time() - total_start_time
        logger.info(f"Parallel batch processing completed in {total_time:.2f}s")
        
        return results
    
    def save_results(self, results: List[Dict[str, Any]], output_file: str):
        """Save processing results to JSON file"""
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            logger.info(f"Results saved to: {output_file}")
        except Exception as e:
            logger.error(f"Error saving results: {e}")
    
    def generate_summary(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate summary statistics from results"""
        total_images = len(results)
        successful = sum(1 for r in results if r['status'] == 'completed')
        failed = total_images - successful
        
        total_time = sum(r.get('processing_time', 0) for r in results)
        avg_time = total_time / total_images if total_images > 0 else 0
        
        summary = {
            'total_images': total_images,
            'successful': successful,
            'failed': failed,
            'success_rate': successful / total_images if total_images > 0 else 0,
            'total_processing_time': total_time,
            'average_processing_time': avg_time,
            'failed_images': [
                r['image_path'] for r in results if r['status'] == 'failed'
            ]
        }
        
        return summary


def load_image_prompt_pairs(config_file: str) -> List[Tuple[str, str]]:
    """
    Load image-prompt pairs from configuration file
    
    Expected format:
    [
        {
            "image_path": "path/to/image1.jpg",
            "text_prompt": "Description for image 1"
        },
        {
            "image_path": "path/to/image2.jpg", 
            "text_prompt": "Description for image 2"
        }
    ]
    """
    try:
        with open(config_file, 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        pairs = []
        for item in config:
            image_path = item.get('image_path')
            text_prompt = item.get('text_prompt')
            
            if not image_path or not text_prompt:
                logger.warning(f"Skipping invalid item: {item}")
                continue
                
            # Validate image path exists
            if not os.path.exists(image_path):
                logger.warning(f"Image not found: {image_path}")
                continue
                
            pairs.append((image_path, text_prompt))
        
        logger.info(f"Loaded {len(pairs)} valid image-prompt pairs from {config_file}")
        return pairs
        
    except Exception as e:
        logger.error(f"Error loading config file: {e}")
        return []


def create_sample_config(output_file: str = "sample_config.json"):
    """Create a sample configuration file"""
    sample_config = [
        {
            "image_path": "path/to/image1.jpg",
            "text_prompt": "A medical brain MRI scan showing a well-circumscribed, extra-axial mass suggestive of a meningioma tumor."
        },
        {
            "image_path": "path/to/image2.jpg", 
            "text_prompt": "A chest X-ray image showing pulmonary nodules in the right lung field."
        },
        {
            "image_path": "path/to/image3.jpg",
            "text_prompt": "An abdominal CT scan revealing a liver lesion with irregular margins."
        }
    ]
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(sample_config, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Sample configuration created: {output_file}")


def main():
    """Main function for batch processing"""
    parser = argparse.ArgumentParser(
        description='MedCLIP-SAMv2 Batch Processing',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process images using config file
  python batch_run.py --config config.json --output results/
  
  # Process with custom settings
  python batch_run.py --config config.json --output results/ --device cuda --parallel
  
  # Create sample config
  python batch_run.py --create-sample-config
        """
    )
    
    parser.add_argument(
        '--config',
        type=str,
        help='Path to JSON configuration file with image-prompt pairs'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default='batch_results',
        help='Output directory for results (default: batch_results)'
    )
    
    parser.add_argument(
        '--device',
        type=str,
        default=None,
        choices=['cuda', 'cpu'],
        help='Device to run models on (auto-detect if not specified)'
    )
    
    parser.add_argument(
        '--parallel',
        action='store_true',
        help='Enable parallel processing (use with caution for GPU)'
    )
    
    parser.add_argument(
        '--max-workers',
        type=int,
        default=1,
        help='Maximum number of parallel workers (default: 1)'
    )
    
    parser.add_argument(
        '--create-sample-config',
        action='store_true',
        help='Create a sample configuration file'
    )
    
    parser.add_argument(
        '--save-results',
        type=str,
        default='batch_results.json',
        help='File to save processing results (default: batch_results.json)'
    )
    
    args = parser.parse_args()
    
    if args.create_sample_config:
        create_sample_config()
        return
    
    if not args.config:
        logger.error("Please provide a configuration file with --config")
        return
    
    if not os.path.exists(args.config):
        logger.error(f"Configuration file not found: {args.config}")
        return
    
    try:
        # Load image-prompt pairs
        image_prompt_pairs = load_image_prompt_pairs(args.config)
        
        if not image_prompt_pairs:
            logger.error("No valid image-prompt pairs found")
            return
        
        # Initialize batch processor
        processor = BatchProcessor(
            device=args.device,
            max_workers=args.max_workers
        )
        
        # Process batch
        results = processor.process_batch(
            image_prompt_pairs=image_prompt_pairs,
            output_dir=args.output,
            use_parallel=args.parallel
        )
        
        # Generate and display summary
        summary = processor.generate_summary(results)
        
        print("\n" + "="*60)
        print("BATCH PROCESSING SUMMARY")
        print("="*60)
        print(f"Total Images: {summary['total_images']}")
        print(f"Successful: {summary['successful']}")
        print(f"Failed: {summary['failed']}")
        print(f"Success Rate: {summary['success_rate']:.2%}")
        print(f"Total Time: {summary['total_processing_time']:.2f}s")
        print(f"Average Time per Image: {summary['average_processing_time']:.2f}s")
        
        if summary['failed_images']:
            print(f"\nFailed Images:")
            for img in summary['failed_images']:
                print(f"  - {img}")
        
        print("="*60)
        
        # Save results
        processor.save_results(results, args.save_results)
        
        # Save summary
        summary_file = args.save_results.replace('.json', '_summary.json')
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Summary saved to: {summary_file}")
        
    except Exception as e:
        logger.error(f"Batch processing failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 