import os
import cv2
import torch
from ultralytics import YOLO

def run_inference(image_path, model_path=None):
    """
    Runs object detection inference on a single image using Apple Silicon GPU (MPS).
    Tracks object counts for all categories and saves the annotated output.
    """
    # 1. Automatically locate the best weights if not provided
    if model_path is None:
        model_path = "runs/detect/local/visdrone/weights/best.pt"

    print(f"Loading custom model weights from: {model_path}")
    
    # 2. Check hardware accelerator availability (Apple Silicon MPS)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Using device for acceleration: {device.upper()}")

    # 3. Load the trained YOLO26 model
    model = YOLO(model_path)

    print(f"Starting inference on image: {image_path}")
    
    # 4. Run prediction optimized for a single photo
    results = model.predict(
        source=image_path,
        device=device,
        save=True,          # Save annotated images automatically
        conf=0.25,          # Confidence threshold (ignore detections below 25%)
        iou=0.45,           # Intersection Over Union threshold for NMS
        exist_ok=True,      # Merge results into the same directory instead of creating predict2, predict3, etc.
        project="local",
        name="predict_images"
    )

    # 5. Process photo telemetry and counts
    result = results[0]
    counts = {}
    
    # Parse detected bounding boxes
    for box in result.boxes:
        class_id = int(box.cls[0])
        class_name = result.names[class_id]
        counts[class_name] = counts.get(class_name, 0) + 1

    # Format clean telemetry log for console output
    count_str = ", ".join([f"{k}: {v}" for k, v in counts.items()])
    if count_str:
        print(f"\nDetections -> {count_str}")
    else:
        print("\nNo targets detected.")

    print(f"\n🎉 Inference completed successfully!")
    print("Annotated output file has been saved to your local directory: local/predict_images/")

if __name__ == "__main__":
    # Example usage for a validation image (change filename to any existing one)
    sample_image = "data/datasets/VisDrone/images/val/0000001_08414_d_0000013.jpg"
    
    if os.path.exists(sample_image):
        run_inference(image_path=sample_image)
    else:
        print(f"Error: Sample image not found at '{sample_image}'.")
        print("Please verify your data/datasets/VisDrone/images/val/ path.")
