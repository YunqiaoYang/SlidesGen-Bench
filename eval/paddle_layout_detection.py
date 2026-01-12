"""
Batch Layout Detection Script for PPT Slide Images
Recursively processes all slide_images folders and saves detection results.
Optimized with parallel processing for faster execution.
"""

import os
# Disable model source connectivity check for faster startup
os.environ['PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK'] = "1"

from paddleocr import LayoutDetection
from glob import glob
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from multiprocessing import cpu_count
import traceback


def get_all_slide_folders(base_dir: str) -> list:
    """
    Recursively find all slide_images folders under base_dir.
    """
    slide_folders = []
    
    for root, dirs, files in os.walk(base_dir):
        if os.path.basename(root) == "slide_images":
            # Get the parent folder (topic folder)
            parent_dir = os.path.dirname(root)
            # Get relative path for display
            rel_path = os.path.relpath(parent_dir, base_dir)
            
            slide_folders.append({
                "rel_path": rel_path,
                "slide_images_path": root,
                "detection_output_path": os.path.join(parent_dir, "detection")
            })
    
    return sorted(slide_folders, key=lambda x: x["rel_path"])


def get_image_files(folder_path: str) -> list:
    """
    Get all image files (png, jpg, jpeg) from a folder.
    """
    extensions = ["*.png", "*.jpg", "*.jpeg", "*.PNG", "*.JPG", "*.JPEG"]
    image_files = []
    for ext in extensions:
        image_files.extend(glob(os.path.join(folder_path, ext)))
    return sorted(image_files)


def process_folder_sequential(folder_info, model_dir, skip_existing):
    """
    Process a single folder with all its images sequentially.
    Model is loaded once and reused for all images.
    
    Args:
        folder_info: Dictionary with folder information
        model_dir: Path to the model directory
        skip_existing: If True, skip images that already have detection results
    
    Returns:
        Tuple of (processed_count, skipped_count, error_count)
    """
    rel_path = folder_info["rel_path"]
    slide_images_path = folder_info["slide_images_path"]
    detection_output_path = folder_info["detection_output_path"]
    
    # Get all images in this folder
    image_files = get_image_files(slide_images_path)
    
    if not image_files:
        return (0, 0, 0)
    
    # Create output directory
    os.makedirs(detection_output_path, exist_ok=True)
    
    # Filter out images that already have results if skip_existing is True
    if skip_existing:
        images_to_process = []
        skipped_count = 0
        for img_path in image_files:
            img_stem = os.path.splitext(os.path.basename(img_path))[0]
            json_path = os.path.join(detection_output_path, f"{img_stem}.json")
            if not os.path.exists(json_path):
                images_to_process.append(img_path)
            else:
                skipped_count += 1
        
        if not images_to_process:
            return (0, skipped_count, 0)
    else:
        images_to_process = image_files
        skipped_count = 0
    
    processed_count = 0
    error_count = 0
    
    # Load model once for this folder
    model = LayoutDetection(model_dir=model_dir)
    
    # Process each image sequentially with the same model
    for img_path in tqdm(images_to_process, desc=f"  {rel_path}", leave=False):
        img_name = os.path.basename(img_path)
        img_stem = os.path.splitext(img_name)[0]
        
        try:
            # Run detection
            output = model.predict(img_path, batch_size=1, layout_nms=True)
            
            # Save results
            for res in output:
                # Save visualization image
                res.save_to_img(save_path=detection_output_path)
                
                # Save JSON result with image-specific name
                json_path = os.path.join(detection_output_path, f"{img_stem}.json")
                res.save_to_json(save_path=json_path)
            
            processed_count += 1
            
        except Exception as e:
            error_count += 1
            print(f"\n    Error processing {img_name}: {e}")
            traceback.print_exc()
    
    return (processed_count, skipped_count, error_count)


def run_batch_detection(base_dir: str, model_dir: str, skip_existing: bool = True, 
                       parallel_folders: int = 3):
    """
    Run layout detection on all slide images with parallel processing.
    Uses thread-level parallelism to process multiple folders simultaneously.
    Each folder processes its images sequentially with one model instance.
    
    Args:
        base_dir: Base directory to search for slide_images folders
        model_dir: Path to the model directory
        skip_existing: If True, skip images that already have detection results
        parallel_folders: Number of folders to process in parallel
    """
    # Get all slide folders
    slide_folders = get_all_slide_folders(base_dir)
    print(f"Found {len(slide_folders)} topic folders to process.")
    print(f"Processing {parallel_folders} folders in parallel (each folder loads model once).\n")
    
    total_skipped = 0
    total_processed = 0
    total_errors = 0
    
    # Use ThreadPoolExecutor to process multiple folders in parallel
    # Each thread will load the model once and process all images in that folder
    with ThreadPoolExecutor(max_workers=parallel_folders) as executor:
        # Submit all folder processing tasks
        future_to_folder = {
            executor.submit(process_folder_sequential, folder_info, model_dir, skip_existing): folder_info
            for folder_info in slide_folders
        }
        
        # Process completed tasks with progress bar
        with tqdm(total=len(future_to_folder), desc="Processing folders") as pbar:
            for future in as_completed(future_to_folder):
                folder_info = future_to_folder[future]
                try:
                    processed, skipped, errors = future.result()
                    total_processed += processed
                    total_skipped += skipped
                    total_errors += errors
                    
                    if processed > 0 or errors > 0:
                        rel_path = folder_info["rel_path"]
                        detection_output_path = folder_info["detection_output_path"]
                        print(f"\n[{rel_path}] Processed: {processed}, Skipped: {skipped}, Errors: {errors}")
                        print(f"  Results saved to: {detection_output_path}")
                        
                except Exception as e:
                    print(f"\nError processing folder {folder_info['rel_path']}: {e}")
                    traceback.print_exc()
                
                pbar.update(1)
    
    print("\n" + "="*50)
    print("Batch processing completed!")
    print(f"Total processed: {total_processed}")
    print(f"Total skipped: {total_skipped}")
    print(f"Total errors: {total_errors}")
    print("="*50)


def main():
    # Configuration
    base_dir = "./examples/Slides"
    model_dir = os.path.join(os.path.dirname(__file__), "PP-DocLayout_plus-L_infer")
    
    # Parallel processing settings
    parallel_folders = 4  # Number of folders to process simultaneously
    
    # Verify paths
    if not os.path.exists(base_dir):
        print(f"Error: Base directory not found: {base_dir}")
        return
    
    if not os.path.exists(model_dir):
        print(f"Error: Model directory not found: {model_dir}")
        return
    
    print("="*50)
    print("Layout Detection Batch Processing (Parallel)")
    print("="*50)
    print(f"Base Directory: {base_dir}")
    print(f"Model Directory: {model_dir}")
    print(f"Skip Existing: True")
    print(f"Parallel Folders: {parallel_folders}")
    print(f"Strategy: Each folder loads model once, processes images sequentially")
    print("="*50 + "\n")
    
    run_batch_detection(base_dir, model_dir, skip_existing=True, 
                       parallel_folders=parallel_folders)


if __name__ == "__main__":
    main()
