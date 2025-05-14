import cv2
import time
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

def initialize_captioning_model():
    """Initialize the BLIP model for scene captioning"""
    print("[Captioning] Loading BLIP model...")
    
    try:
        from transformers import BlipProcessor, BlipForConditionalGeneration
        from PIL import Image
        import torch
        
        processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = BlipForConditionalGeneration.from_pretrained(
            "Salesforce/blip-image-captioning-base"
        ).to(device)
        
        print(f"[Captioning] BLIP model loaded successfully on {device}.")
        return processor, model
    except Exception as e:
        print(f"[Captioning] Error loading BLIP model: {e}")
        return None, None

def describe_scene(frame, processor, model):
    """Generate caption for the given frame using BLIP model"""
    try:
        from PIL import Image
        # Convert OpenCV BGR to PIL RGB format
        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        
        # Process with BLIP
        inputs = processor(image, return_tensors="pt").to(model.device)
        out = model.generate(**inputs, max_length=50)
        caption = processor.decode(out[0], skip_special_tokens=True)
        
        return caption.strip()
    except Exception as e:
        print(f"[Captioning] Error generating caption: {e}")
        return "Caption generation failed."

def run_captioning_mode(speak_callback, speech_running):
    """Run the scene captioning assistant"""
    # Initialize model
    processor, model = initialize_captioning_model()
    if processor is None or model is None:
        print("[Captioning] Failed to initialize models.")
        speak_callback("Scene captioning model could not be loaded.")
        return "exit"
    
    # Initialize camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[Captioning] Error: Cannot access camera.")
        speak_callback("Camera not available for captioning mode.")
        return "exit"
    
    speak_callback("Scene captioning mode started. I will describe what I see.")
    
    last_caption = ""
    last_caption_time = 0
    caption_interval = 5  # Generate a new caption every 5 seconds
    
    print("\nKeyboard shortcuts:")
    print("  q - Return to main menu")
    print("  n - Switch to navigation mode")
    print("  s - Switch to sign detection mode")
    print("  m - Switch to currency detection mode")
    print("  v - Toggle voice commands")
    
    next_mode = None
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("[Captioning] Camera frame acquisition failed.")
            time.sleep(0.5)
            continue

        try:
            current_time = time.time()
            
            # Only generate a new caption every 'interval' seconds
            if current_time - last_caption_time > caption_interval:
                caption = describe_scene(frame, processor, model)
                
                # Only announce if caption has changed
                if caption != last_caption:
                    speak_callback(f"Scene: {caption}")
                    last_caption = caption
                
                last_caption_time = current_time

            # Display the frame with current caption
            cv2.putText(frame, f"Scene: {last_caption}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            
            # Add controls overlay and display
            frame = add_controls_overlay(frame, speech_running)
            cv2.imshow("Scene Captioning Assistant", frame)
            
        except Exception as e:
            print(f"[Captioning] Error in processing: {e}")
        
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