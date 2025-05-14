import cv2
import numpy as np
import time
import os
from ultralytics import YOLO

# Import from our modules
from ..utils.ui import add_controls_overlay, handle_common_keys
from ..recognition.voice_commands import check_voice_commands

# Add to the beginning of each mode function

def run_xxx_mode(speak_callback, speech_running):
    """Run the xxx mode"""
    
    # Test speech functionality
    try:
        speak_callback("Starting mode")
        time.sleep(0.5)
    except Exception as e:
        print(f"Speech error at mode start: {e}")
        # Try to recover by using direct print instead
        print("Starting mode (speech error occurred)")

def initialize_navigation_models():
    """Initialize the YOLO and MiDaS models"""
    print("[Navigation] Loading models...")
    
    # Get the models directory path
    models_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "models")
    
    # Load YOLOv8 model
    yolo_model_path = os.path.join(models_dir, "yolov8m.pt")
    yolo_model = YOLO(yolo_model_path)
    
    # Load MiDaS model
    midas_model_path = os.path.join(models_dir, "midas_small.onnx")
    midas = cv2.dnn.readNet(midas_model_path)
    
    print("[Navigation] Models loaded successfully.")
    return yolo_model, midas

def get_depth_map(frame, midas):
    """Get depth map from frame using MiDaS model"""
    try:
        blob = cv2.dnn.blobFromImage(frame, 1/255.0, (256, 256), swapRB=True, crop=False)
        midas.setInput(blob)
        depth = midas.forward()[0, :, :]
        depth = cv2.resize(depth, (frame.shape[1], frame.shape[0]))
        return depth
    except Exception as e:
        print(f"[Navigation] Error creating depth map: {e}")
        return np.zeros((frame.shape[0], frame.shape[1]))

def analyze_navigation(objects, width):
    """Analyze objects to determine navigation instructions"""
    left = right = center = 0
    for obj in objects:
        x_center = (obj['box'][0] + obj['box'][2]) // 2
        if x_center < width // 3:
            left += 1
        elif x_center > 2 * width // 3:
            right += 1
        else:
            center += 1
    
    if center > 0:
        return "Obstacle ahead. Please stop."
    elif left > right:
        return "Objects on your left. Move to the right."
    elif right > left:
        return "Objects on your right. Move to the left."
    return "Clear path ahead. You can go forward."

def run_navigation_mode(speak_callback, speech_running):
    """Run the navigation assistant"""
    # Initialize models
    yolo_model, midas = initialize_navigation_models()
    coco_classes = yolo_model.names
    
    # Initialize camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[Navigation] Error: Cannot access camera.")
        speak_callback("Camera not available for navigation mode.")
        return "exit"
    
    speak_callback("Navigation mode started. I will help you navigate.")
    
    prev_instruction = ""
    instruction_time = 0
    
    print("\nKeyboard shortcuts:")
    print("  q - Return to main menu")
    print("  c - Switch to captioning mode")
    print("  s - Switch to sign detection mode")
    print("  m - Switch to currency detection mode")
    print("  v - Toggle voice commands")
    
    next_mode = None
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("[Navigation] Camera frame acquisition failed.")
            time.sleep(0.5)
            continue

        try:
            # Process with YOLOv8
            results = yolo_model(frame)[0]
            
            # Get depth information
            depth_map = get_depth_map(frame, midas)
            objects_detected = []

            # Process detections
            for result in results.boxes:
                cls_id = int(result.cls)
                cls_name = coco_classes[cls_id]
                conf = float(result.conf)
                
                # Skip low confidence detections
                if conf < 0.5:
                    continue
                    
                box = result.xyxy[0].cpu().numpy().astype(int)
                x1, y1, x2, y2 = box
                
                # Get average depth in region
                region_depth = depth_map[y1:y2, x1:x2]
                if region_depth.size > 0:
                    avg_depth = np.mean(region_depth)
                else:
                    avg_depth = 0

                objects_detected.append({
                    'name': cls_name,
                    'conf': conf,
                    'box': box,
                    'depth': avg_depth
                })

                # Draw on frame
                label = f"{cls_name} {conf:.2f}"
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

            # Get navigation instruction
            instruction = analyze_navigation(objects_detected, frame.shape[1])
            cv2.putText(frame, instruction, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            current_time = time.time()
            # Speak only if instruction changed or every 5 seconds
            if (instruction != prev_instruction or current_time - instruction_time > 5) and objects_detected:
                names = ", ".join(set([obj['name'] for obj in objects_detected]))
                speak_callback(f"I see {names}. {instruction}")
                prev_instruction = instruction
                instruction_time = current_time
            elif not objects_detected and current_time - instruction_time > 5:
                speak_callback("No objects detected. Path is clear.")
                instruction_time = current_time

            # Add controls overlay and display
            frame = add_controls_overlay(frame, speech_running)
            cv2.imshow("Navigation Assistant", frame)
            
        except Exception as e:
            print(f"[Navigation] Error in processing: {e}")
        
        # Check for key presses for mode switching
        key = cv2.waitKey(1) & 0xFF
        next_mode = handle_common_keys(key)
        
        # Check for voice commands
        if next_mode is None:
            next_mode = check_voice_commands()
        
        if next_mode:
            if next_mode != "exit":
                print(f"Switching to {next_mode} mode.")
                speak_callback(f"Switching to {next_mode} mode.")
            else:
                print("Returning to main menu.")
                speak_callback("Returning to main menu.")
            break
            
    # Clean up
    cap.release()
    cv2.destroyAllWindows()
    
    return next_mode