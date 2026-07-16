import os
import cv2
import torch
from ultralytics import YOLO

def process_video_sequence(sequence_folder, fps=30):
    """
    Processes a specific folder of sequential drone images using Apple Silicon GPU (MPS).
    Saves the final compiled MP4 file named exactly after the input directory.
    """
    # 1. Pinned project path to the last trained model weights
    model_path = "runs/detect/local/visdrone/weights/best.pt"
    
    print(f"Loading custom weights from: {model_path}")
    model = YOLO(model_path)
    
    # 2. Activate Apple Silicon hardware acceleration (Metal Performance Shaders)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Hardware acceleration: {device.upper()}")

    # 3. Clean path parsing to dynamically extract the folder name
    sequence_path = os.path.abspath(sequence_folder)
    folder_name = os.path.basename(sequence_path)
    output_name = f"{folder_name}.mp4"

    # 4. Get sorted list of images to maintain the exact temporal frame sequence
    frames = sorted([f for f in os.listdir(sequence_path) if f.lower().endswith(('.jpg', '.jpeg'))])
    if not frames:
        raise FileNotFoundError(f"Error: No image frames found inside '{sequence_path}'.")

    print(f"Processing sequence '{folder_name}' ({len(frames)} frames total)...")

    # 5. Automatically read dimensions from the first frame to setup container width/height
    first_frame_path = os.path.join(sequence_path, frames[0])
    first_frame = cv2.imread(first_frame_path)
    height, width, _ = first_frame.shape

    # 6. Initialize the OpenCV VideoWriter stream with Mac native 'mp4v' container codec
    save_dir = "runs/detect/local/predict_videos"
    os.makedirs(save_dir, exist_ok=True)
    save_video_path = os.path.join(save_dir, output_name)
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(save_video_path, fourcc, fps, (width, height))

    # 7. Run frame-by-frame pipeline without saving heavy temporary raw visual plots to SSD
    for frame_name in frames:
        frame_path = os.path.join(sequence_path, frame_name)
        results = model.predict(source=frame_path, device=device, verbose=False, conf=0.25)
        result = results[0]

        # Draw native YOLO bounding boxes on top of the raw image data
        annotated_frame = result.plot()

        # Parse live objects stream statistics
        counts = {}
        for box in result.boxes:
            class_id = int(box.cls)
            class_name = result.names[class_id]
            counts[class_name] = counts.get(class_name, 0) + 1

        # Draw a translucent telemetry dashboard overlay box in the top-left corner
        cv2.rectangle(annotated_frame, (10, 10), (280, 80), (20, 20, 20), -1)
        
        car_count = counts.get('car', 0)
        ped_count = counts.get('pedestrian', 0)
        
        cv2.putText(annotated_frame, f"Live Telemetry Stream", (20, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(annotated_frame, f"Cars: {car_count} | Pedestrians: {ped_count}", (20, 55), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

        # Write frame buffer directly into the video file stream container
        video_writer.write(annotated_frame)

    # 8. Release the writer stream and finalize container headers
    video_writer.release()
    print(f"\n🎉 Process completed! Output file saved to: {save_video_path}")

if __name__ == "__main__":
    # Define the target folder path directly (Change 'uav0000182_00000_v' to your exact directory name)
    target_sequence_folder = "data/datasets/VisDrone/videos/VisDrone2019-VID-val/sequences/uav0000339_00001_v"
    
    process_video_sequence(sequence_folder=target_sequence_folder)
