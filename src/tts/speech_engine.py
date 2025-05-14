import pyttsx3
import threading
import time

# Global variables for TTS
tts_engine = None
tts_lock = threading.Lock()
speaking = threading.Event()
last_spoken = ""
last_spoken_time = 0
speech_enabled = True  # New flag to control if speech is enabled

def get_tts_engine():
    """Get or create global TTS engine with thread safety"""
    global tts_engine
    with tts_lock:
        if tts_engine is None:
            try:
                tts_engine = pyttsx3.init()
                tts_engine.setProperty('rate', 150)
            except Exception as e:
                print(f"Error initializing TTS engine: {e}")
    return tts_engine

def speak(text):
    """Thread-safe speaking function that avoids the 'run loop already started' error"""
    global last_spoken, last_spoken_time
    
    # Skip if speech is disabled
    if not speech_enabled:
        print(f"Speech output (disabled): {text}")
        return
    
    # Don't repeat the same phrase too quickly
    if text == last_spoken and time.time() - last_spoken_time < 3:
        return
    
    # Don't try to speak if we're already speaking
    if speaking.is_set():
        print(f"Already speaking, skipping: {text}")
        return
    
    def speak_thread():
        global last_spoken, last_spoken_time
        
        try:
            speaking.set()
            engine = get_tts_engine()
            if not engine:  # Check if engine initialization failed
                print(f"Cannot speak: {text} (TTS engine not available)")
                return
                
            engine.say(text)
            
            # This is the part that throws "run loop already started"
            # We put proper safeguards around it
            try:
                engine.runAndWait()
            except RuntimeError as e:
                # Handle "run loop already started" error
                if "run loop already started" in str(e):
                    print("TTS run loop issue detected")
                    # Try to reinitialize the engine
                    with tts_lock:
                        global tts_engine
                        try:
                            tts_engine = None
                            tts_engine = pyttsx3.init()
                            tts_engine.setProperty('rate', 150)
                            tts_engine.say(text)
                            tts_engine.runAndWait()
                        except Exception as reinit_error:
                            print(f"Failed to reinitialize TTS: {reinit_error}")
                else:
                    print(f"TTS Error: {e}")
                    
            last_spoken = text
            last_spoken_time = time.time()
        except Exception as e:
            print(f"TTS Error: {e}")
        finally:
            speaking.clear()
    
    # Start a new thread to do the speaking
    threading.Thread(target=speak_thread, daemon=True).start()

def toggle_speech():
    """Toggle speech on/off"""
    global speech_enabled
    speech_enabled = not speech_enabled
    return speech_enabled

def cleanup_tts():
    """Clean up TTS resources"""
    global tts_engine
    
    with tts_lock:
        if tts_engine:
            try:
                # Make sure engine isn't still running
                if speaking.is_set():
                    # Wait for speaking to finish
                    time.sleep(0.5)
                
                # Attempt to stop the engine
                tts_engine.stop()
            except:
                pass
            tts_engine = None