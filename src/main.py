import os
import sys
import time

# Make sure package is importable (for running from command line)
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import modules
from src.tts.speech_engine import speak, cleanup_tts, toggle_speech
from src.recognition.voice_commands import (
    VOSK_AVAILABLE, start_voice_commands, 
    stop_voice_commands, toggle_voice_commands, 
    check_voice_commands
)
from src.modes.navigation import run_navigation_mode
from src.modes.captioning import run_captioning_mode
from src.modes.sign_detection import run_sign_detection_mode
from src.modes.currency_detection import run_currency_detection_mode

# Global state
speech_running = False

def check_models_directory():
    """Verify models directory structure and required files"""
    # [existing code]

def ask_for_mode():
    """Ask user which mode they want to use"""
    speak("Which mode would you like to use? Navigation, Captioning, Sign Detection, Currency Detection, or Voice Command?")
    return input("\nEnter mode (nav/cap/sign/curr/voice/exit): ").strip().lower()

def handle_mode_selection(command):
    """Handle mode selection with proper feedback"""
    global speech_running  # Move this to the top of the function
    
    if command == "nav":
        speak("Starting navigation mode.")
        time.sleep(0.5)  # Give time for speech to finish
        return run_navigation_mode(speak, speech_running)
    elif command == "cap":
        speak("Starting captioning mode.")
        time.sleep(0.5)
        return run_captioning_mode(speak, speech_running)
    elif command == "sign":
        speak("Starting sign detection mode.")
        time.sleep(0.5)
        return run_sign_detection_mode(speak, speech_running)
    elif command == "curr":
        speak("Starting currency detection mode.")
        time.sleep(0.5)
        return run_currency_detection_mode(speak, speech_running)
    elif command == "speech":
        is_enabled = toggle_speech()
        if is_enabled:
            print("Speech output enabled")
            speak("Speech output enabled")
        else:
            print("Speech output disabled")
        return None
    elif command == "voice":
        toggle_voice_commands(speak)
        speech_running = not speech_running
        if speech_running:
            speak("Voice commands are now active. You can speak commands.")
        else:
            speak("Voice commands are now inactive.")
        return None
    elif command == "exit":
        speak("Exiting assistant. Goodbye.")
        print("Exiting assistant...")
        return "exit"
    else:
        speak("I didn't understand that command. Please try again with nav, cap, sign, curr, voice, or exit.")
        print("Unknown command. Try nav, cap, sign, curr, voice, or exit.")
        return None

def main():
    """Main application entry point"""
    global speech_running
    
    # Check models
    check_models_directory()
    
    print("\n===== AI Assistant for the Visually Impaired =====")
    print("Commands:")
    print("  nav     - Switch to navigation mode")
    print("  cap     - Switch to scene captioning mode")
    print("  sign    - Switch to sign detection mode")
    print("  curr    - Switch to currency detection mode")
    print("  voice   - Toggle voice command mode (default: OFF)")
    print("  speech  - Toggle speech output (default: ON)")
    print("  exit    - Exit the assistant")
    print("=============================================\n")

    # Welcome message
    speak("Welcome to AI Assistant for the Visually Impaired.")
    time.sleep(1)
    
    # Ask user if they want to enable voice commands at startup instead of auto-enabling
    print("\nWould you like to enable voice commands? (yes/no)")
    speak("Would you like to enable voice commands?")
    
    voice_choice = input("Enable voice commands? (yes/no): ").strip().lower()
    if voice_choice in ["yes", "y"]:
        if VOSK_AVAILABLE:
            speech_running = True
            start_voice_commands(speak)
            speak("Voice commands activated. You can now speak your mode choice.")
        else:
            speak("Voice command system is not available. Using keyboard input only.")
    else:
        speak("Using keyboard input for commands.")
    
    # Main interaction loop    
    try:
        while True:
            # Ask user which mode they want
            mode = None
            
            # First check voice commands if enabled
            if speech_running:
                print("Listening for voice command... (or type a command)")
                for _ in range(5):  # Check for 5 seconds max
                    mode = check_voice_commands()
                    if mode:
                        print(f"Voice command detected: {mode}")
                        break
                    time.sleep(1)
            
            # If no voice command, ask for typed input
            if not mode:
                mode = ask_for_mode()
            
            # Execute the selected mode
            result = handle_mode_selection(mode)
            
            # Check if we should exit
            if result == "exit":
                break
            elif result == "voice":
                toggle_voice_commands(speak)
                speech_running = not speech_running
                
    except KeyboardInterrupt:
        print("\nInterrupted. Exiting gracefully...")
    except Exception as e:
        print(f"Error in main loop: {e}")
    finally:
        # Stop voice commands
        if speech_running:
            stop_voice_commands()
        
        # Clean up TTS resources
        cleanup_tts()
        
        # Final cleanup
        print("Goodbye!")

if __name__ == "__main__":
    main()