import subprocess
import os
import json
import sys
import time
import psutil
from pathlib import Path

def check_process_status(process_name):
    """Check if a specific process is running"""
    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            if process_name.lower() in ' '.join(proc.info['cmdline']).lower():
                return True, proc.info['pid']
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return False, None

def run_medclip_classification(image_path, output_dir):
    """Run MedCLIP classification model"""
    print("Running MedCLIP Classification...")
    print(f"   Working directory: {os.getcwd()}")
    print(f"   Image path: {image_path}")
    print(f"   Conda environment: medclip")
    print(f"   Script: model/MedCLIP/run.py")
    
    try:
        print("   Starting MedCLIP process...")
        start_time = time.time()
        
        # Chạy MedCLIP classification với conda environment 'medclip'
        print("   Running command: conda run -n medclip python model/MedCLIP/run.py")
        result = subprocess.run([
            "conda", "run", "-n", "medclip",
            "python", "model/MedCLIP/run.py"
        ], 
        capture_output=True, 
        text=True, 
        cwd=os.getcwd(),
        check=True,
        timeout=300  # 5 phút timeout
        )
        
        end_time = time.time()
        duration = end_time - start_time
        
        print("MedCLIP Classification completed successfully!")
        print(f"   Execution time: {duration:.2f} seconds")
        print("   Output:")
        print("   " + "="*50)
        # In từng dòng output với indent
        for line in result.stdout.split('\n'):
            if line.strip():  # Chỉ in dòng không rỗng
                print(f"   {line}")
        print("   " + "="*50)
        
        # Lưu kết quả classification
        classification_output = os.path.join(output_dir, "classification")
        os.makedirs(classification_output, exist_ok=True)
        
        # Tạo file kết quả đơn giản
        classification_result = {
            "model": "MedCLIP",
            "status": "completed",
            "output": result.stdout
        }
        
        with open(os.path.join(classification_output, "result.json"), "w") as f:
            json.dump(classification_result, f, indent=4)
            
        return classification_result
        
    except subprocess.CalledProcessError as e:
        print(f"Error running MedCLIP Classification: {e}")
        print(f"Error output: {e.stderr}")
        return {"model": "MedCLIP", "status": "failed", "error": e.stderr}
    except subprocess.TimeoutExpired as e:
        print(f"Timeout running MedCLIP Classification (5 minutes)")
        print(f"Process may still be running. Check manually.")
        return {"model": "MedCLIP", "status": "timeout", "error": "Process timed out"}

def run_medclip_samv2(image_path, output_dir):
    """Run MedCLIP-SAMv2 segmentation model"""
    print("Running MedCLIP-SAMv2 Segmentation...")
    print(f"   Working directory: {os.getcwd()}")
    print(f"   Image path: {image_path}")
    
    try:
        print("   Starting MedCLIP-SAMv2 process...")
        start_time = time.time()
        
        # Chạy MedCLIP-SAMv2 với conda environment 'medclip-sam'
        result = subprocess.run([
            "conda", "run", "-n", "medclip-sam",
            "python", "model/MedCLIP-SAMv2/run.py"
        ], 
        capture_output=True, 
        text=True, 
        cwd=os.getcwd(),
        check=True
        )
        
        end_time = time.time()
        duration = end_time - start_time
        
        print("MedCLIP-SAMv2 Segmentation completed successfully!")
        print(f"   Execution time: {duration:.2f} seconds")
        print("   Output:")
        print("   " + "="*50)
        # In từng dòng output với indent
        for line in result.stdout.split('\n'):
            if line.strip():  # Chỉ in dòng không rỗng
                print(f"   {line}")
        print("   " + "="*50)
        
        # Lưu kết quả segmentation
        segmentation_output = os.path.join(output_dir, "segmentation")
        os.makedirs(segmentation_output, exist_ok=True)
        
        # Tạo file kết quả đơn giản
        segmentation_result = {
            "model": "MedCLIP-SAMv2",
            "status": "completed",
            "output": result.stdout
        }
        
        with open(os.path.join(segmentation_output, "result.json"), "w") as f:
            json.dump(segmentation_result, f, indent=4)
            
        return segmentation_result
        
    except subprocess.CalledProcessError as e:
        print(f"Error running MedCLIP-SAMv2: {e}")
        print(f"Error output: {e.stderr}")
        return {"model": "MedCLIP-SAMv2", "status": "failed", "error": e.stderr}

def run_models(image_path=None):
    """Main function to run both models"""
    print("Starting Medical Image Analysis Pipeline")
    print("=" * 60)
    
    # Tạo thư mục kết quả
    output_dir = "results"
    os.makedirs(output_dir, exist_ok=True)
    
    # Kiểm tra conda environments
    print("Checking conda environments...")
    try:
        # Kiểm tra environment medclip
        subprocess.run(["conda", "env", "list"], capture_output=True, text=True, check=True)
        print("Conda environments check completed")
    except subprocess.CalledProcessError as e:
        print(f"Error checking conda environments: {e}")
        return
    
    # Chạy MedCLIP Classification
    classification_result = run_medclip_classification(image_path, output_dir)
    
    print("\n" + "-" * 60)
    
    # Chạy MedCLIP-SAMv2 Segmentation
    segmentation_result = run_medclip_samv2(image_path, output_dir)
    
    # Kết hợp kết quả
    final_result = {
        "classification": classification_result,
        "segmentation": segmentation_result,
        "timestamp": str(Path().cwd()),
        "image_path": image_path
    }
    
    # Lưu kết quả tổng hợp
    final_output_path = os.path.join(output_dir, "final_result.json")
    with open(final_output_path, "w", encoding='utf-8') as f:
        json.dump(final_result, f, indent=4, ensure_ascii=False)
    
    print("\n" + "=" * 60)
    print("Processing completed!")
    print(f"Results saved in '{output_dir}' directory")
    print(f"Final result: {final_output_path}")
    
    # Hiển thị tóm tắt kết quả
    print("\nSummary:")
    print(f"   Classification: {classification_result['status']}")
    print(f"   Segmentation: {segmentation_result['status']}")

if __name__ == "__main__":
    # Có thể truyền đường dẫn ảnh qua command line argument
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
    else:
        # Sử dụng ảnh mặc định từ MedCLIP
        image_path = "model/MedCLIP/example_data/view1_frontal.jpg"
    
    print(f"Using image: {image_path}")
    
    # Kiểm tra file ảnh tồn tại
    if not os.path.exists(image_path):
        print(f"Image file not found: {image_path}")
        print("Please provide a valid image path as argument")
        sys.exit(1)
    
    run_models(image_path)