import cv2
import numpy as np
from deep_sort_realtime.deepsort_tracker import DeepSort
import time
from ultralytics import YOLO
from collections import defaultdict

# Model and tracking parameters
CONFIDENCE_THRESHOLD = 0.7
FRAME_SKIP = 1  # Process every frame for better detection reliability
MAX_TRACKING_DURATION = 45  # Maximum tracking duration in seconds

# Video configuration
video_path = "/Users/amansubash/Downloads/ttriple.mp4"
output_path = "/Users/amansubash/Downloads/output_5sec.mp4"

# Load YOLOv8 model
model_path = "/Users/amansubash/Downloads/triples_weights.pt"
model = YOLO(model_path)

# Initialize DeepSORT tracker with enhanced tracking parameters
tracker = DeepSort(
    max_age=500,
    n_init=1,
    nms_max_overlap=0.7,
    max_cosine_distance=0.5,
    nn_budget=300,
    override_track_class=None,
    embedder="mobilenet",
    half=True,
    bgr=True
)

# Global tracking state
unique_ids = set()
track_start_times = {}  # Store start time for each track
track_durations = {}    # Store tracking duration for each track
currently_tracked_id = None  # Store the ID of the currently tracked rider

def get_detections(frame):
    """Get triple riding detections using YOLOv8 model"""
    results = model(frame, conf=CONFIDENCE_THRESHOLD)[0]
    detections = []

    for r in results.boxes.data:
        x1, y1, x2, y2, conf, class_id = r
        if int(class_id) == 0:  # Triple riding class
            w = x2 - x1
            h = y2 - y1
            
            # Additional checks to confirm it's triple riding
            aspect_ratio = w / h
            area = w * h
            
            # Balanced area requirements
            min_area = (frame.shape[0] * frame.shape[1]) * 0.01
            max_area = (frame.shape[0] * frame.shape[1]) * 0.5
            
            # More flexible aspect ratio for different viewing angles
            is_valid_size = min_area < area < max_area
            is_valid_ratio = 0.5 < aspect_ratio < 2.5
            
            if is_valid_size and is_valid_ratio:
                detections.append(([int(x1), int(y1), int(w), int(h)], float(conf), "triple_riding"))
    
    return detections

def is_leaving_frame(box, frame_shape, margin=10):
    """Check if the box is at the edge of the frame"""
    height, width = frame_shape[:2]
    x1, y1, x2, y2 = box
    
    return (x1 <= margin or y1 <= margin or 
            x2 >= width - margin or y2 >= height - margin)

def draw_info_overlay(frame, info_dict):
    """Draw a semi-transparent overlay with tracking information"""
    height, width = frame.shape[:2]
    
    # Create semi-transparent overlay for the top bar
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (width, 80), (0, 0, 0), -1)
    
    # Create gradient effect for better visibility
    gradient = np.linspace(0.8, 0.2, 80)
    for i in range(80):
        cv2.line(overlay, (0, i), (width, i), (0, 0, 0), 1)
    
    # Blend overlay with original frame
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
    
    # Add text with enhanced styling
    font = cv2.FONT_HERSHEY_DUPLEX
    total_count = info_dict['total_unique']
    
    # Draw total count with large font
    count_text = f"{total_count}"
    text_size = cv2.getTextSize(count_text, font, 2.5, 3)[0]
    text_x = width - text_size[0] - 20
    
    # Draw count numbers
    cv2.putText(frame, count_text, (text_x, 55), 
                font, 2.5, (0, 255, 0), 3)
    
    # Draw labels
    cv2.putText(frame, "Total Triple Riding Cases:", (20, 55), 
                font, 1.2, (255, 255, 255), 2)
    
    return frame

def main():
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    frame_id = 0
    tracks = []
    current_time = time.time()
    global currently_tracked_id
    
    try:
        print("🏍️ Starting sequential triple riding detection (5-second per rider)...")
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("✅ Finished processing video.")
                break

            current_time = time.time()
            process_frame = frame_id % FRAME_SKIP == 0
            
            if process_frame:
                try:
                    # Get detections from model
                    detections = get_detections(frame)
                    
                    # Update tracker
                    tracks = tracker.update_tracks(detections, frame=frame)

                    # Check if we need to switch to a new target
                    if currently_tracked_id is None or (
                        currently_tracked_id in track_durations and 
                        track_durations[currently_tracked_id] > MAX_TRACKING_DURATION
                    ):
                        # Look for a new target
                        for track in tracks:
                            if not track.is_confirmed():
                                continue
                            
                            track_id = track.track_id
                            if track_id not in track_durations or track_durations[track_id] <= MAX_TRACKING_DURATION:
                                currently_tracked_id = track_id
                                if track_id not in track_start_times:
                                    track_start_times[track_id] = current_time
                                    track_durations[track_id] = 0
                                    unique_ids.add(track_id)
                                    print(f"⚠️ Locking onto new rider! ID #{track_id}")
                                break

                    # Process tracks
                    for track in tracks:
                        if not track.is_confirmed():
                            continue

                        track_id = track.track_id
                        
                        # Only process the currently tracked ID
                        if track_id != currently_tracked_id:
                            continue

                        ltrb = track.to_ltrb()
                        confidence = track.det_conf if track.det_conf is not None else 0.0
                        x1, y1, x2, y2 = map(int, ltrb)

                        # Calculate tracking duration
                        track_durations[track_id] = current_time - track_start_times[track_id]

                        # Skip if tracking duration exceeds limit
                        if track_durations[track_id] > MAX_TRACKING_DURATION:
                            continue

                        # Calculate color based on remaining time
                        remaining_time = MAX_TRACKING_DURATION - track_durations[track_id]
                        if remaining_time <= 1.0:  # Last second
                            color = (0, 165, 255)  # Orange
                            thickness = 2
                        else:
                            color = (255, 0, 255)  # Purple
                            thickness = 3

                        # Draw bounding box
                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

                        # Draw label with ID, confidence, and remaining time
                        label = f"ID: {track_id} ({confidence:.2f}) {remaining_time:.1f}s"
                        (text_w, text_h), _ = cv2.getTextSize(
                            label, cv2.FONT_HERSHEY_DUPLEX, 0.6, 2
                        )
                        
                        # Draw label background
                        cv2.rectangle(
                            frame,
                            (x1, y1-text_h-8),
                            (x1 + text_w, y1),
                            color,
                            -1
                        )
                        cv2.putText(frame, label,
                                  (x1, y1-5), cv2.FONT_HERSHEY_DUPLEX,
                                  0.6, (255, 255, 255), 2)

                except Exception as e:
                    print(f"⚠️ Error processing frame {frame_id}: {str(e)}")
                    continue

            # Clean up expired tracks
            if currently_tracked_id in track_durations and track_durations[currently_tracked_id] > MAX_TRACKING_DURATION:
                print(f"✅ Finished tracking rider #{currently_tracked_id}, looking for next target...")
                currently_tracked_id = None

            # Draw info overlay
            info_dict = {
                'total_unique': len(unique_ids),
                'frame': frame_id
            }
            frame = draw_info_overlay(frame, info_dict)

            # Write frame and show
            out.write(frame)
            cv2.imshow("Triple Riding Detection (Sequential 5-sec)", frame)
            cv2.setWindowProperty("Triple Riding Detection (Sequential 5-sec)", 
                                cv2.WND_PROP_TOPMOST, 1)
            
            if cv2.waitKey(1) & 0xFF == ord("q"):
                print("🛑 Interrupted by user.")
                break

            frame_id += 1

    except Exception as e:
        print(f"❌ Critical error: {str(e)}")
        import traceback
        traceback.print_exc()
    finally:
        cap.release()
        out.release()
        cv2.destroyAllWindows()

        print("\n📊 Final Statistics")
        print(f"Frames Processed: {frame_id}")
        print(f"Total Unique Triple Riding Cases: {len(unique_ids)}")
        print(f"📁 Output saved to: {output_path}")

if __name__ == "__main__":
    main()