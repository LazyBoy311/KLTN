import subprocess
import os
import sys

def test_medclip_direct():
    """Test running MedCLIP directly"""
    print("🧪 Testing MedCLIP directly...")
    
    # Thử chạy trực tiếp
    try:
        print("1️⃣ Testing direct Python execution...")
        result = subprocess.run([
            "python", "model/MedCLIP/run.py"
        ], 
        capture_output=True, 
        text=True, 
        cwd=os.getcwd(),
        timeout=60
        )
        print("✅ Direct execution successful!")
        print("Output:", result.stdout)
        return True
    except Exception as e:
        print(f"❌ Direct execution failed: {e}")
    
    # Thử với conda run nhưng không capture output
    try:
        print("2️⃣ Testing conda run without capture...")
        result = subprocess.run([
            "conda", "run", "-n", "medclip",
            "python", "model/MedCLIP/run.py"
        ], 
        cwd=os.getcwd(),
        timeout=60
        )
        print("✅ Conda run without capture successful!")
        return True
    except Exception as e:
        print(f"❌ Conda run without capture failed: {e}")
    
    # Thử kiểm tra environment
    try:
        print("3️⃣ Testing conda environment...")
        result = subprocess.run([
            "conda", "run", "-n", "medclip",
            "python", "-c", "print('Environment test successful')"
        ], 
        capture_output=True, 
        text=True, 
        timeout=30
        )
        print("✅ Environment test successful!")
        print("Output:", result.stdout)
        return True
    except Exception as e:
        print(f"❌ Environment test failed: {e}")
    
    return False

if __name__ == "__main__":
    test_medclip_direct() 