import os
import threading
import queue
import json
import time

# Try to import vosk for speech recognition
try:
    import vosk
    import pyaudio
    VOSK_AVAILABLE = True
except ImportError:
    print("Warning: Vosk not available. Install with: pip install vosk pyaudio")
    VOSK_AVAILABLE = False

# Voice command setup
# Update at the top level of file
voice_command_queue = queue.Queue()
speech_thread = None
speech_running = False
auto_start = False  # Don't start automatically

def initialize_vosk():
    """Initialize Vosk speech recognition model"""
    try:
        if not VOSK_AVAILABLE:
            return None
            
        # Look for the model in the models directory
        model_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 
                              "models", "vosk-model-small-en-us-0.15")
        
        if not os.path.exists(model_dir):
            print(f"Vosk model not found at {model_dir}")
            print("Please download from https://alphacephei.com/vosk/models")
            print("Extract the model to:", os.path.abspath(model_dir))
            return None
            
        model = vosk.Model(model_dir)
        print("Voice command recognition initialized.")
        return model
    except Exception as e:
        print(f"Error initializing speech recognition: {e}")
        return None

def process_voice_commands(speak_callback):
    """Process voice commands in a separate thread"""
    global speech_running
    
    try:
        if not VOSK_AVAILABLE:
            print("Vosk not available for voice commands.")
            return
            
        model = initialize_vosk()
        if model is None:
            print("Could not initialize speech recognition model.")
            return
            
        # Initialize PyAudio
        p = pyaudio.PyAudio()
        rec = vosk.KaldiRecognizer(model, 16000)
        
        # Open microphone stream
        stream = p.open(format=pyaudio.paInt16, channels=1, rate=16000,
                        input=True, frames_per_buffer=8000)
        stream.start_stream()
        
        print("Voice command system active. Try saying: 'navigation', 'captioning', 'signs', 'currency', or 'exit'")
        speak_callback("Voice commands activated.")
        
        while speech_running:
            try:
                data = stream.read(4000, exception_on_overflow=False)
                
                if rec.AcceptWaveform(data):
                    result = json.loads(rec.Result())
                    if "text" in result and result["text"].strip() != "":
                        command = result["text"].lower()
                        print(f"Heard: '{command}'")
                        
                        # Process command
                        if "exit" in command or "quit" in command or "menu" in command:
                            voice_command_queue.put("exit")
                        elif "nav" in command or "navi" in command or "navigate" in command or "navigation" in command:
                            voice_command_queue.put("nav")
                        elif "cap" in command or "scene" in command or "describe" in command or "caption" in command:
                            voice_command_queue.put("cap")
                        elif "sign" in command or "road" in command:
                            voice_command_queue.put("sign")
                        elif "currency" in command or "money" in command or "cash" in command:
                            voice_command_queue.put("curr")
            except Exception as e:
                print(f"Error processing audio: {e}")
                time.sleep(0.1)  # Brief pause on error to avoid CPU spinning
                        
        # Clean up
        stream.stop_stream()
        stream.close()
        p.terminate()
        
    except Exception as e:
        print(f"Error in voice command processing: {e}")
    finally:
        print("Voice command system stopped.")

def start_voice_commands(speak_callback):
    """Start the voice command recognition thread"""
    global speech_thread, speech_running
    
    if speech_thread is None or not speech_thread.is_alive():
        speech_running = True
        speech_thread = threading.Thread(target=process_voice_commands, args=(speak_callback,), daemon=True)
        speech_thread.start()
        return True
    return False

def stop_voice_commands():
    """Stop the voice command recognition thread"""
    global speech_running
    
    if speech_running:
        speech_running = False
        return True
    return False

def toggle_voice_commands(speak_callback):
    """Toggle voice command recognition on/off"""
    global speech_running
    
    if speech_running:
        result = stop_voice_commands()
        if result:
            speak_callback("Voice commands deactivated.")
            print("Voice commands turned off.")
    else:
        result = start_voice_commands(speak_callback)
        if result:
            speak_callback("Voice commands activated. You can now give voice instructions.")
            print("Voice commands turned on.")

def check_voice_commands():
    """Check if there are any voice commands in the queue"""
    try:
        command = voice_command_queue.get_nowait()
        return command
    except queue.Empty:
        return None