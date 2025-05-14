import cv2
import time
import numpy as np
import os

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

def run_currency_detection_mode(speak_callback, speech_running):
    """Run the currency detection assistant"""
    print("[Currency Detection] Loading model...")
    
    try:
        from tensorflow.keras.models import load_model
        
        # Class labels for currency detection
        class_labels = ['10', '100', '20', '200', '2000', '50', '500']
        img_size = (224, 224)
        
        # Get model path
        models_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "models")
        model_path = os.path.join(models_dir, "custom_cnn_model.h5")
        
        # Load currency model
        model = load_model(model_path)
        
        def detect_currency(frame):
            # Preprocess image
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, img_size)
            img = img.astype(np.float32) / 255.0
            img = np.expand_dims(img, axis=0)
            
            # Get predictions
            preds = model.predict(img)
            class_idx = np.argmax(preds)
            confidence = float(np.max(preds))
            
            # Get label
            label = class_labels[class_idx] if class_idx < len(class_labels) else 'Unknown'
            return label, confidence
            
        print("[Currency Detection] Model loaded successfully.")
    except Exception as e:
        print(f"[Currency Detection] Error loading model: {e}")
        speak_callback("Error loading currency detection model.")
        return "exit"
    
    # Initialize camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[Currency Detection] Error: Cannot access camera.")
        speak_callback("Camera not available for currency detection.")
        return "exit"
    
    speak_callback("Currency detection mode active. Please show currency notes to the camera.")
    
    prev_label = None
    last_announcement_time = 0
    min_confidence = 0.7  # Minimum confidence threshold
    
    print("\nKeyboard shortcuts:")
    print("  q - Return to main menu")
    print("  n - Switch to navigation mode")
    print("  c - Switch to captioning mode")
    print("  s - Switch to sign detection mode")
    print("  v - Toggle voice commands")
    
    next_mode = None
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("[Currency Detection] Camera frame acquisition failed.")
            time.sleep(0.5)
            continue

        try:
            # Process current frame
            label, confidence = detect_currency(frame)
            
            current_time = time.time()
            
            # Only announce if confidence is above threshold
            if confidence > min_confidence:
                output_text = f"{label} rupees (Confidence: {confidence:.2f})"
                
                # Only announce if label changed or time passed
                if (label != prev_label or current_time - last_announcement_time > 5) and label != 'Unknown':
                    speak_callback(f"{label} rupees detected")
                    prev_label = label
                    last_announcement_time = current_time
            else:
                output_text = "No currency detected"
                # If nothing detected for a while, reset prev_label
                if current_time - last_announcement_time > 10:
                    prev_label = None
            
            # Display information on frame
            cv2.putText(frame, output_text, (20, 40), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Add controls overlay and display
            frame = add_controls_overlay(frame, speech_running)
            cv2.imshow('Currency Detection', frame)
            
        except Exception as e:
            print(f"[Currency Detection] Error in processing: {e}")
        
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