import cv2

def add_controls_overlay(frame, speech_running=False):
    """Add control information overlay to frame"""
    h, w = frame.shape[:2]
    
    # Create overlay for controls at bottom of frame
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, h-70), (w, h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
    
    # Add text for controls
    cv2.putText(frame, "Controls: q=exit, n=nav, c=cap, s=sign, m=curr, v=voice", 
               (10, h-45), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    # Add voice command status
    status = "ON" if speech_running else "OFF"
    cv2.putText(frame, f"Voice commands: {status}", 
               (10, h-20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 
               (0, 255, 0) if speech_running else (0, 0, 255), 1)
    
    return frame

def handle_common_keys(key):
    """Handle common keyboard shortcuts for mode switching"""
    if key == ord('q'):
        return "exit"  # Exit to main menu
    elif key == ord('n'):
        return "nav"   # Navigation mode
    elif key == ord('c'):
        return "cap"   # Captioning mode
    elif key == ord('s'):
        return "sign"  # Sign detection mode
    elif key == ord('m'):
        return "curr"  # Currency detection mode
    elif key == ord('v'):
        # Toggle voice command mode (handled separately)
        return "voice"
    return None        # No mode change