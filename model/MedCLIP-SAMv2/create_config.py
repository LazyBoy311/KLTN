"""
Helper script to create configuration files for batch processing
"""

import os
import json
import argparse
from pathlib import Path
from typing import List, Dict, Any


def get_image_files(directory: str, extensions: List[str] = None) -> List[str]:
    """
    Get all image files from a directory
    
    Args:
        directory: Directory to search
        extensions: List of file extensions to include (default: common image formats)
    
    Returns:
        List of image file paths
    """
    if extensions is None:
        extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']
    
    image_files = []
    directory_path = Path(directory)
    
    if not directory_path.exists():
        print(f"Directory not found: {directory}")
        return image_files
    
    for ext in extensions:
        # Use case-insensitive pattern matching
        image_files.extend(directory_path.glob(f"*{ext}"))
        image_files.extend(directory_path.glob(f"*{ext.upper()}"))
    
    # Convert to strings, remove duplicates, and sort
    image_files = list(set([str(f) for f in image_files]))
    image_files.sort()
    
    return image_files


def create_config_from_directory(
    image_dir: str,
    output_file: str,
    prompt_template: str = None,
    custom_prompts: Dict[str, str] = None
) -> None:
    """
    Create configuration file from directory of images
    
    Args:
        image_dir: Directory containing images
        output_file: Output configuration file path
        prompt_template: Template prompt to use for all images
        custom_prompts: Dictionary mapping image names to custom prompts
    """
    image_files = get_image_files(image_dir)
    
    if not image_files:
        print(f"No image files found in {image_dir}")
        return
    
    config = []
    
    for image_path in image_files:
        image_name = Path(image_path).stem
        
        # Use custom prompt if available, otherwise use template
        if custom_prompts and image_name in custom_prompts:
            text_prompt = custom_prompts[image_name]
        elif prompt_template:
            text_prompt = prompt_template
        else:
            # Default medical image prompt
            text_prompt = "A medical image showing anatomical structures and potential abnormalities."
        
        config.append({
            "image_path": image_path,
            "text_prompt": text_prompt
        })
    
    # Save configuration
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    
    print(f"Created configuration file: {output_file}")
    print(f"Added {len(config)} images to configuration")


def create_config_from_list(
    image_list: List[str],
    output_file: str,
    prompt_template: str = None,
    custom_prompts: Dict[str, str] = None
) -> None:
    """
    Create configuration file from list of image paths
    
    Args:
        image_list: List of image file paths
        output_file: Output configuration file path
        prompt_template: Template prompt to use for all images
        custom_prompts: Dictionary mapping image names to custom prompts
    """
    config = []
    
    for image_path in image_list:
        if not os.path.exists(image_path):
            print(f"Warning: Image not found: {image_path}")
            continue
        
        image_name = Path(image_path).stem
        
        # Use custom prompt if available, otherwise use template
        if custom_prompts and image_name in custom_prompts:
            text_prompt = custom_prompts[image_name]
        elif prompt_template:
            text_prompt = prompt_template
        else:
            # Default medical image prompt
            text_prompt = "A medical image showing anatomical structures and potential abnormalities."
        
        config.append({
            "image_path": image_path,
            "text_prompt": text_prompt
        })
    
    # Save configuration
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    
    print(f"Created configuration file: {output_file}")
    print(f"Added {len(config)} images to configuration")


def load_custom_prompts(prompts_file: str) -> Dict[str, str]:
    """
    Load custom prompts from various file formats
    
    Supported formats:
    1. JSON: {"image_name": "prompt"}
    2. CSV: image_name,prompt
    3. TXT: image_name: prompt (one per line)
    4. JSON Array: [{"image_name": "name", "prompt": "text"}]
    
    Args:
        prompts_file: Path to prompts file
        
    Returns:
        Dictionary mapping image names to prompts
    """
    try:
        file_ext = Path(prompts_file).suffix.lower()
        
        if file_ext == '.json':
            return _load_json_prompts(prompts_file)
        elif file_ext == '.csv':
            return _load_csv_prompts(prompts_file)
        elif file_ext == '.txt':
            return _load_txt_prompts(prompts_file)
        else:
            # Try to auto-detect format
            return _auto_detect_prompts_format(prompts_file)
            
    except Exception as e:
        print(f"Error loading prompts file: {e}")
        return {}


def _load_json_prompts(prompts_file: str) -> Dict[str, str]:
    """Load prompts from JSON file"""
    with open(prompts_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Handle different JSON formats
    if isinstance(data, dict):
        # Format: {"image_name": "prompt"}
        return data
    elif isinstance(data, list):
        # Format: [{"image_name": "name", "prompt": "text"}]
        prompts = {}
        for item in data:
            if isinstance(item, dict):
                image_name = item.get('image_name') or item.get('name') or item.get('filename')
                prompt = item.get('prompt') or item.get('text') or item.get('description')
                if image_name and prompt:
                    prompts[image_name] = prompt
        return prompts
    else:
        raise ValueError("Unsupported JSON format")


def _load_csv_prompts(prompts_file: str) -> Dict[str, str]:
    """Load prompts from CSV file"""
    import csv
    
    prompts = {}
    
    # First try to detect the separator
    with open(prompts_file, 'r', encoding='utf-8') as f:
        first_line = f.readline().strip()
    
    # Check for common separators
    separators = ['|', ',', ';', '\t']
    detected_separator = None
    
    for sep in separators:
        if sep in first_line:
            parts = first_line.split(sep)
            if len(parts) >= 2:
                detected_separator = sep
                break
    
    if detected_separator is None:
        # Fallback to comma
        detected_separator = ','
    
    print(f"Detected separator: '{detected_separator}'")
    
    with open(prompts_file, 'r', encoding='utf-8') as f:
        reader = csv.reader(f, delimiter=detected_separator)
        
        # Skip header if exists
        first_row = next(reader, None)
        if first_row and len(first_row) >= 2:
            # Check if first row is header
            if not first_row[0].lower() in ['image_name', 'name', 'filename', 'image']:
                # First row is data
                prompt_text = first_row[1].strip('"')  # Remove quotes
                prompts[first_row[0]] = prompt_text
            
            # Process remaining rows
            for row in reader:
                if len(row) >= 2:
                    prompt_text = row[1].strip('"')  # Remove quotes
                    prompts[row[0]] = prompt_text
    
    return prompts


def _load_txt_prompts(prompts_file: str) -> Dict[str, str]:
    """Load prompts from TXT file"""
    prompts = {}
    
    with open(prompts_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            
            # Try different separators
            for separator in [':', '|', ';', '\t']:
                if separator in line:
                    parts = line.split(separator, 1)
                    if len(parts) == 2:
                        image_name = parts[0].strip()
                        prompt = parts[1].strip()
                        if image_name and prompt:
                            prompts[image_name] = prompt
                            break
            else:
                print(f"Warning: Line {line_num} format not recognized: {line}")
    
    return prompts


def _auto_detect_prompts_format(prompts_file: str) -> Dict[str, str]:
    """Auto-detect file format and load prompts"""
    with open(prompts_file, 'r', encoding='utf-8') as f:
        content = f.read().strip()
    
    # Try JSON first
    try:
        data = json.loads(content)
        return _load_json_prompts_from_content(data)
    except json.JSONDecodeError:
        pass
    
    # Try CSV format
    if ',' in content and '\n' in content:
        try:
            return _load_csv_prompts(prompts_file)
        except:
            pass
    
    # Try TXT format
    try:
        return _load_txt_prompts(prompts_file)
    except:
        pass
    
    raise ValueError(f"Could not detect format for file: {prompts_file}")


def _load_json_prompts_from_content(data) -> Dict[str, str]:
    """Load prompts from JSON content"""
    if isinstance(data, dict):
        return data
    elif isinstance(data, list):
        prompts = {}
        for item in data:
            if isinstance(item, dict):
                image_name = item.get('image_name') or item.get('name') or item.get('filename')
                prompt = item.get('prompt') or item.get('text') or item.get('description')
                if image_name and prompt:
                    prompts[image_name] = prompt
        return prompts
    else:
        raise ValueError("Unsupported JSON format")


def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description='Create configuration files for MedCLIP-SAMv2 batch processing',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Create config from directory with default prompt
  python create_config.py --image-dir ./images --output config.json
  
  # Create config with custom template prompt
  python create_config.py --image-dir ./images --output config.json --prompt "Brain MRI scan"
  
  # Create config with custom prompts file (various formats supported)
  python create_config.py --image-dir ./images --output config.json --custom-prompts prompts.json
  python create_config.py --image-dir ./images --output config.json --custom-prompts prompts.csv
  python create_config.py --image-dir ./images --output config.json --custom-prompts prompts.txt
  
  # Create config from list of images
  python create_config.py --image-list image1.jpg image2.jpg --output config.json

Supported prompt file formats:
  1. JSON: {"image_name": "prompt"} or [{"image_name": "name", "prompt": "text"}]
  2. CSV: image_name,prompt (with or without header)
  3. TXT: image_name: prompt (one per line, supports :, |, ;, tab separators)
        """
    )
    
    parser.add_argument(
        '--image-dir',
        type=str,
        help='Directory containing images'
    )
    
    parser.add_argument(
        '--image-list',
        nargs='+',
        help='List of image file paths'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        required=True,
        help='Output configuration file path'
    )
    
    parser.add_argument(
        '--prompt',
        type=str,
        help='Template prompt to use for all images'
    )
    
    parser.add_argument(
        '--custom-prompts',
        type=str,
        help='File containing custom prompts for specific images (supports JSON, CSV, TXT formats)'
    )
    
    args = parser.parse_args()
    
    # Load custom prompts if provided
    custom_prompts = {}
    if args.custom_prompts:
        custom_prompts = load_custom_prompts(args.custom_prompts)
    
    # Create configuration
    if args.image_dir:
        create_config_from_directory(
            image_dir=args.image_dir,
            output_file=args.output,
            prompt_template=args.prompt,
            custom_prompts=custom_prompts
        )
    elif args.image_list:
        create_config_from_list(
            image_list=args.image_list,
            output_file=args.output,
            prompt_template=args.prompt,
            custom_prompts=custom_prompts
        )
    else:
        print("Please provide either --image-dir or --image-list")
        return


if __name__ == "__main__":
    main() 