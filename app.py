import streamlit as st
import subprocess
import os
import json
import time
import sys
from PIL import Image
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import tempfile
import shutil

def main():
    st.set_page_config(
        page_title="Medical Image Analysis",
        page_icon="🏥",
        layout="wide"
    )
    
    st.title("🏥 Medical Image Analysis Pipeline")
    st.markdown("---")
    
    # Sidebar for file upload and settings
    st.sidebar.header("📁 Upload Image")
    uploaded_file = st.sidebar.file_uploader(
        "Choose an image file", 
        type=['png', 'jpg', 'jpeg']
    )
    
    # Custom prompts input
    st.sidebar.header("🔤 Custom Prompts")
    st.sidebar.markdown("Add custom prompts to improve classification accuracy")
    
    custom_prompts = st.sidebar.text_area(
        "Enter custom prompts (one per line):",
        placeholder="pneumonia\ntumor\nnormal\nfracture",
        help="Enter medical conditions or findings to look for in the image"
    )
    
    # Parse custom prompts
    prompt_list = None
    if custom_prompts.strip():
        prompt_list = [prompt.strip() for prompt in custom_prompts.split('\n') if prompt.strip()]
        st.sidebar.success(f"✅ {len(prompt_list)} custom prompts loaded")
    
    # Analysis settings
    st.sidebar.header("⚙️ Analysis Settings")
    save_results = st.sidebar.checkbox("Save results to file", value=True)
    output_filename = st.sidebar.text_input(
        "Output filename:", 
        value="analysis_results.json",
        help="Filename for saving results"
    )
    
    # Debug section
    st.sidebar.header("🐛 Debug Options")
    enable_logging = st.sidebar.checkbox(
        "Enable Terminal Logging", 
        value=True,
        help="Show real-time logs in terminal"
    )

    
    # Main content area
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("🖼️ Input Image")
        if uploaded_file is not None:
            # Display uploaded image
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_column_width=True)
            
            # Image info
            st.info(f"📊 Image size: {image.size[0]} x {image.size[1]} pixels")
            
            # Save uploaded file temporarily
            temp_path = save_uploaded_file(uploaded_file)
            
            # Analysis button
            if st.button("🔬 Run Analysis", type="primary"):
                with st.spinner("Running analysis..."):
                    results = run_analysis(temp_path, prompt_list, save_results, output_filename, enable_logging)
                    display_results(results, col2)
                    
                    # Cleanup temp file
                    cleanup_temp_file(temp_path)
        else:
            st.info("Please upload an image to start analysis")
    
    with col2:
        st.header("📊 Results")
        if 'results' not in st.session_state:
            st.info("Results will appear here after analysis")

def save_uploaded_file(uploaded_file):
    """Save uploaded file to temporary location"""
    temp_dir = tempfile.mkdtemp()
    temp_path = os.path.join(temp_dir, uploaded_file.name)
    
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    return temp_path

def cleanup_temp_file(temp_path):
    """Clean up temporary file"""
    try:
        if os.path.exists(temp_path):
            os.remove(temp_path)
            temp_dir = os.path.dirname(temp_path)
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
    except Exception as e:
        st.warning(f"Could not clean up temporary file: {e}")

def run_analysis(image_path, custom_prompts=None, save_to_file=True, output_filename="analysis_results.json", enable_logging=True):
    """Run the analysis pipeline"""
    try:
        # Create results directory
        output_dir = "results"
        os.makedirs(output_dir, exist_ok=True)
        
        # Progress bar
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Step 1: MedCLIP Classification
        status_text.text("Running MedCLIP Classification...")
        progress_bar.progress(25)
        
        classification_result = run_medclip_classification(
            image_path, 
            custom_prompts, 
            save_to_file, 
            output_filename,
            enable_logging
        )
        # classification_result = {"status": "completed", "data": {"confidence_scores": [{"class": "pneumonia", "confidence": 0.95}]}}
        # Step 2: MedCLIP-SAMv2 Segmentation
        status_text.text("Running MedCLIP-SAMv2 Segmentation...")
        progress_bar.progress(75)
        
        # Use custom prompts as text for segmentation if available
        custom_text = None
        if custom_prompts:
            custom_text = " ".join(custom_prompts[:3])  # Use first 3 prompts as text
        
        
        # Check if image_path exists before running segmentation
        if not os.path.exists(image_path):
            st.error(f"❌ Image path does not exist: {image_path}")
            segmentation_result = {"status": "failed", "error": f"Image file not found: {image_path}"}
        else:
            st.info(f"✅ Image path verified: {image_path}")

            segmentation_result = run_medclip_samv2(image_path, output_dir, custom_text, enable_logging)
        
        # Complete
        progress_bar.progress(100)
        status_text.text("Analysis completed!")
        
        return {
            "classification": classification_result,
            "segmentation": segmentation_result
        }
        
    except Exception as e:
        st.error(f"Error during analysis: {str(e)}")
        return None

def run_medclip_classification(image_path, custom_prompts=None, save_to_file=True, output_filename="analysis_results.json", enable_logging=True):
    """Run MedCLIP classification with new command line interface"""
    try:
        # Build command with arguments
        cmd = [
            "conda", "run", "-n", "medclip",
            "python", "model/MedCLIP/run.py"
        ]
        
        # Add image path if provided
        if image_path:
            cmd.extend(["--image", image_path])
        
        # Add custom prompts if provided
        if custom_prompts:
            cmd.extend(["--prompts"] + custom_prompts)
        
        # Add output filename if provided
        if output_filename:
            cmd.extend(["--output", output_filename])
        
        # Add no-save flag if save_to_file is False
        if not save_to_file:
            cmd.append("--no-save")
        
        st.info(f"Running command: {' '.join(cmd)}")
        
        # Run subprocess with real-time output to terminal
        if enable_logging:
            print(f"🚀 Running MedCLIP classification...")
            sys.stdout.flush()
        
        # Run subprocess
        if enable_logging:
            # Run with output to terminal
            process = subprocess.Popen(
                cmd,
                stdout=None,
                stderr=None,
                cwd=os.getcwd()
            )
        else:
            # Run without output
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                cwd=os.getcwd()
            )
        
        # Wait for process to complete with timeout
        try:
            return_code = process.wait(timeout=600)  # 10 minutes timeout
        except subprocess.TimeoutExpired:
            print("[ERROR] MedCLIP process timed out after 10 minutes!")
            process.kill()
            return {"status": "timeout", "error": "Process timed out after 10 minutes"}
        
        # Check for errors
        if return_code != 0:
            st.error(f"MedCLIP process failed with return code {return_code}")
            return {"status": "failed", "error": f"Process failed with return code {return_code}"}
        
        # Parse output to extract classification results
        classification_data = parse_medclip_output([], output_filename)
        
        return {
            "status": "completed",
            "data": classification_data,
            "raw_output": "Classification completed successfully",
            "command": ' '.join(cmd)
        }
        
    except subprocess.TimeoutExpired:
        return {"status": "timeout", "error": "Process timed out"}
    except Exception as e:
        return {"status": "failed", "error": str(e)}

def run_medclip_samv2(image_path, output_dir, custom_text=None, enable_logging=True):
    """Run MedCLIP-SAMv2 segmentation using conda activate"""
    try:
        # Build command using conda activate
        cmd = f"conda activate medclip-sam && python model/MedCLIP-SAMv2/run.py"
        
        # Test if we can run a simple conda command first
        if enable_logging:
            print(f"🔍 Testing conda command...")
            try:
                test_result = subprocess.run(["conda", "--version"], capture_output=True, text=True, timeout=10)
                if test_result.returncode == 0:
                    print(f"✅ Conda is available: {test_result.stdout.strip()}")
                else:
                    print(f"❌ Conda test failed: {test_result.stderr}")
            except Exception as e:
                print(f"❌ Conda test error: {e}")
            sys.stdout.flush()
        
        # Add image path if provided
        if image_path:
            cmd += f" --image {image_path}"
        
        # Add custom text if provided
        if custom_text:
            cmd += f" --text '{custom_text}'"
        
        st.info(f"Running MedCLIP-SAMv2 command: {cmd}")
        
        # Debug: Check if file exists and conda environment
        if enable_logging:
            print(f"🚀 Running MedCLIP-SAMv2 segmentation...")
            print(f"🔍 Checking file existence:")
            print(f"   - Image path: {image_path} (exists: {os.path.exists(image_path)})")
            print(f"   - Run.py path: model/MedCLIP-SAMv2/run.py (exists: {os.path.exists('model/MedCLIP-SAMv2/run.py')})")
            print(f"   - Working directory: {os.getcwd()}")
            
            # Check conda environment
            try:
                import subprocess as sp
                result = sp.run(["conda", "env", "list"], capture_output=True, text=True)
                if "medclip-sam" in result.stdout:
                    print(f"✅ Conda environment 'medclip-sam' found")
                else:
                    print(f"❌ Conda environment 'medclip-sam' not found")
                    print(f"Available environments: {result.stdout}")
            except Exception as e:
                print(f"⚠️ Could not check conda: {e}")
            sys.stdout.flush()
        
        # Run subprocess with better error handling
        try:
            if enable_logging:
                # Run with output to terminal using shell=True
                process = subprocess.Popen(
                    cmd,
                    shell=True,
                    stdout=None,
                    stderr=None,
                    cwd=os.getcwd(),
                    env=dict(os.environ, TOKENIZERS_PARALLELISM="false")
                )
            else:
                # Run without output using shell=True
                process = subprocess.Popen(
                    cmd,
                    shell=True,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    cwd=os.getcwd(),
                    env=dict(os.environ, TOKENIZERS_PARALLELISM="false")
                )
        except FileNotFoundError as e:
            error_msg = f"Command not found: conda"
            if enable_logging:
                print(f"[ERROR] {error_msg}")
            return {"status": "failed", "error": error_msg}
        except Exception as e:
            error_msg = f"Failed to start subprocess: {str(e)}"
            if enable_logging:
                print(f"[ERROR] {error_msg}")
            return {"status": "failed", "error": error_msg}
        
        # Wait for process to complete with timeout
        try:
            return_code = process.wait(timeout=600)  # 10 minutes timeout
        except subprocess.TimeoutExpired:
            print("[ERROR] MedCLIP-SAMv2 process timed out after 10 minutes!")
            process.kill()
            return {"status": "timeout", "error": "Process timed out after 10 minutes"}
        
        # Check for errors
        if return_code != 0:
            st.error(f"MedCLIP-SAMv2 process failed with return code {return_code}")
            return {"status": "failed", "error": f"Process failed with return code {return_code}"}
        
        # Check for output files
        output_files = check_samv2_output_files()
        
        return {
            "status": "completed",
            "raw_output": "Segmentation completed successfully",
            "command": cmd,
            "output_files": output_files
        }
        
    except Exception as e:
        return {"status": "failed", "error": str(e)}

def check_samv2_output_files():
    """Check for MedCLIP-SAMv2 output files"""
    output_files = {}
    
    # Check for SAM output
    sam_output_path = "model/MedCLIP-SAMv2/sam_output.png"
    if os.path.exists(sam_output_path):
        output_files["sam_output"] = sam_output_path
    
    # Check for postprocessed map
    postprocessed_path = "model/MedCLIP-SAMv2/postprocessed_map.png"
    if os.path.exists(postprocessed_path):
        output_files["postprocessed_map"] = postprocessed_path
    
    # Check for any other output files in the directory
    samv2_dir = "model/MedCLIP-SAMv2"
    if os.path.exists(samv2_dir):
        for file in os.listdir(samv2_dir):
            if file.endswith(('.png', '.jpg', '.jpeg')):
                file_path = os.path.join(samv2_dir, file)
                if file not in output_files:
                    output_files[file] = file_path
    
    return output_files

def parse_medclip_output(output_lines, output_filename="analysis_results.json"):
    """Parse MedCLIP output to extract classification results"""
    # Try to read from JSON file first
    json_files_to_try = [
        output_filename,
        "medclip_results.json",
        "results/analysis_results.json"
    ]
    
    for json_file in json_files_to_try:
        try:
            if os.path.exists(json_file):
                with open(json_file, "r") as f:
                    data = json.load(f)
                    st.success(f"✅ Successfully loaded results from {json_file}")
                    return data
        except Exception as e:
            st.warning(f"Could not read {json_file}: {e}")
            continue
    
    # Fallback to parsing output lines
    st.warning("Could not find JSON results file, parsing console output...")
    results = {}
    
    for i, line in enumerate(output_lines):
        if "Model Confidence Scores:" in line:
            # Extract confidence scores
            confidence_data = []
            for j in range(i+1, len(output_lines)):
                line = output_lines[j].strip()
                if line.startswith(('1.', '2.', '3.', '4.', '5.', '6.', '7.', '8.', '9.')):
                    parts = line.split()
                    if len(parts) >= 3:
                        try:
                            # Extract class name (everything between number and confidence score)
                            confidence_str = parts[2]
                            confidence = float(confidence_str)
                            
                            # Class name is everything between the number and confidence
                            class_name = ' '.join(parts[1:-1]) if len(parts) > 3 else parts[1]
                            
                            confidence_data.append({
                                "class": class_name,
                                "confidence": confidence
                            })
                        except ValueError:
                            continue
                elif line.startswith('-') or line.startswith('='):
                    break
            
            results["confidence_scores"] = confidence_data
            
        elif "Most Likely Condition:" in line:
            results["most_likely"] = line.split(":")[1].strip()
            
        elif "Confidence:" in line and "Most Likely" not in line:
            try:
                results["top_confidence"] = float(line.split(":")[1].strip())
            except ValueError:
                continue
    
    return results

def display_results(results, col):
    """Display analysis results"""
    if not results:
        col.error("No results to display")
        return
    
    # Classification Results
    if results.get("classification"):
        classification = results["classification"]
        
        if classification["status"] == "completed":
            col.success("✅ Classification completed")
            
            data = classification.get("data", {})
            
            # Display confidence scores as bar chart
            if "confidence_scores" in data and data["confidence_scores"]:
                df = pd.DataFrame(data["confidence_scores"])
                
                # Sort by confidence
                df = df.sort_values('confidence', ascending=False)
                
                fig = px.bar(
                    df, 
                    x="class", 
                    y="confidence",
                    title="Classification Confidence Scores",
                    color="confidence",
                    color_continuous_scale="RdYlGn"
                )
                fig.update_layout(
                    xaxis_tickangle=-45,
                    height=400,
                    showlegend=False
                )
                col.plotly_chart(fig, use_container_width=True)
                
                # Display top 5 results in a table
                col.subheader("📋 Top 5 Results")
                top_5_df = df.head(5)
                col.dataframe(
                    top_5_df[['class', 'confidence']].round(3),
                    use_container_width=True
                )
            
            # Display most likely condition
            if "most_likely" in data:
                col.metric(
                    "🎯 Most Likely Condition", 
                    data["most_likely"],
                    f"{data.get('top_confidence', 0):.3f}"
                )
            
            # Display additional info
            if "total_classes" in data:
                col.info(f"📊 Analyzed {data['total_classes']} different conditions")
            
            # Show command used
            if "command" in classification:
                with col.expander("🔧 Command Details"):
                    st.code(classification["command"])
        else:
            col.error(f"❌ Classification failed: {classification.get('error', 'Unknown error')}")
    
    # Segmentation Results
    if results.get("segmentation"):
        segmentation = results["segmentation"]
        
        if segmentation["status"] == "completed":
            col.success("✅ Segmentation completed")
            
            # Display output files if available
            output_files = segmentation.get("output_files", {})
            
            if output_files:
                col.subheader("🖼️ Segmentation Results")
                
                # Display SAM output
                if "sam_output" in output_files:
                    try:
                        sam_image = Image.open(output_files["sam_output"])
                        col.image(sam_image, caption="SAM Segmentation", use_column_width=True)
                    except Exception as e:
                        col.warning(f"Could not load SAM output: {e}")
                
                # Display postprocessed map
                if "postprocessed_map" in output_files:
                    try:
                        post_image = Image.open(output_files["postprocessed_map"])
                        col.image(post_image, caption="Postprocessed Attention Map", use_column_width=True)
                    except Exception as e:
                        col.warning(f"Could not load postprocessed map: {e}")
                
                # Display other output files
                other_files = {k: v for k, v in output_files.items() 
                              if k not in ["sam_output", "postprocessed_map"]}
                if other_files:
                    col.subheader("📁 Other Output Files")
                    for file_name, file_path in other_files.items():
                        col.info(f"📄 {file_name}: {file_path}")
            else:
                col.warning("⚠️ Segmentation completed but no output files found")
            
            # Show command used
            if "command" in segmentation:
                with col.expander("🔧 Segmentation Command Details"):
                    st.code(segmentation["command"])
        else:
            col.error(f"❌ Segmentation failed: {segmentation.get('error', 'Unknown error')}")

if __name__ == "__main__":
    main() 