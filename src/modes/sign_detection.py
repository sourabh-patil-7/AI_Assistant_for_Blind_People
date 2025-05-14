import cv2
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

def run_sign_detection_mode(speak_callback, speech_running):
    """Run the road sign detection assistant"""
    # Road sign contextual messages
    context_messages = {
        'bus_stop': "Bus stop ahead.",
        'do_not_enter': "Do not enter sign detected. Entry is prohibited.",
        'do_not_stop': "Do not stop sign detected. Stopping is not allowed here.",
        'do_not_turn_l': "No left turn allowed.",
        'do_not_turn_r': "No right turn allowed.",
        'do_not_u_turn': "No U-turn allowed.",
        'enter_left_lane': "Enter left lane.",
        'green_light': "Green light. You may proceed.",
        'left_right_lane': "Left or right lane allowed.",
        'no_parking': "No parking zone.",
        'parking': "Parking area ahead.",
        'ped_crossing': "Pedestrian crossing ahead. Please slow down.",
        'ped_zebra_cross': "Zebra crossing ahead. Watch for pedestrians.",
        'railway_crossing': "Railway crossing ahead. Be cautious.",
        'red_light': "Red light. Please stop.",
        'stop': "Stop sign detected. Please stop.",
        't_intersection_l': "T-intersection to the left ahead.",
        'traffic_light': "Traffic light ahead.",
        'u_turn': "U-turn allowed here.",
        'warning': "Warning sign ahead. Please be careful.",
        'yellow_light': "Yellow light. Prepare to stop."
    }
    
    print("[Sign Detection] Loading model...")
    try:
        # Get model path
        models_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "models")
        model_path = os.path.join(models_dir, "best.pt")
        
        model = YOLO(model_path)  # Load road sign detection model
        class_names = model.names
        print("[Sign Detection] Model loaded successfully.")
    except Exception as e:
        print(f"[Sign Detection] Error loading model: {e}")
        speak_callback("Error loading sign detection model.")
        return "exit"
    
    # Initialize camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[Sign Detection] Error: Cannot access camera.")
        speak_callback("Camera not available for sign detection.")
        return "exit"
    
    speak_callback("Sign detection mode active. I will announce road signs I see.")
    
    prev_labels = set()
    conf_threshold = 0.7
    
    print("\nKeyboard shortcuts:")
    print("  q - Return to main menu")
    print("  n - Switch to navigation mode")
    print("  c - Switch to captioning mode")
    print("  m - Switch to currency detection mode")
    print("  v - Toggle voice commands")
    
    next_mode = None
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("[Sign Detection] Camera frame acquisition failed.")
            time.sleep(0.5)
            continue

        try:
            # Process with YOLO
            results = model(frame)[0]
            detections = results.boxes
            
            current_labels = set()
            valid_detection = False
            
            for box in detections:
                conf = float(box.conf[0])
                if conf < conf_threshold:
                    continue
                    
                valid_detection = True
                cls_id = int(box.cls[0])
                label = class_names[cls_id] if cls_id < len(class_names) else f"Class {cls_id}"
                
                # Draw bounding box
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"{label.replace('_', ' ')} {conf:.2f}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                
                current_labels.add(label)
            
            # Announce new signs
            if valid_detection:
                new_labels = current_labels - prev_labels
                for label in new_labels:
                    label_key = label.lower()
                    message = context_messages.get(label_key, f"{label_key.replace('_', ' ')} detected.")
                    speak_callback(message)
                
                prev_labels = current_labels
            
            # Add controls overlay and display
            frame = add_controls_overlay(frame, speech_running)
            cv2.imshow("Road Sign Detection", frame)
            
        except Exception as e:
            print(f"[Sign Detection] Error in processing: {e}")
        
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