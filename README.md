# AI Assistant for the Visually Impaired

An assistive technology application that uses computer vision and artificial intelligence to help visually impaired individuals navigate their environment, understand their surroundings, and gain more independence.

## Features

- **Navigation Mode**: Detects objects in the environment and provides real-time guidance to help users navigate safely.
- **Scene Captioning Mode**: Describes the surroundings using image captioning technology.
- **Road Sign Detection Mode**: Identifies and announces road signs with contextual explanations.
- **Currency Detection Mode**: Recognizes and announces currency denominations.
- **Voice Command Support**: Enables hands-free control using voice commands.
- **Text-to-Speech Feedback**: Provides audio feedback of all important information.

## Requirements

- Python 3.8+
- Webcam or camera device
- Internet connection (for initial setup)

## Installation

1. Clone the repository:
```
git clone https://github.com/sourabh-patil-7/AI_Assistant_for_Blind_People.git
cd AI_Assistant_for_Blind_People
```

2. Install the required packages:
```
pip install -r requirements.txt
```

3. Download the required models and place them in the `models` directory:
   - `yolov8m.pt`: YOLOv8 model for object detection
   - `midas_small.onnx`: MiDaS depth estimation model
   - `best.pt`: Custom-trained model for road sign detection
   - `custom_cnn_model.h5`: CNN model for currency detection
   - `vosk-model-small-en-us-0.15`: Vosk speech recognition model (optional for voice commands)

## Usage

Run the main application:
```
python src/main.py
```

### Available Commands:
- `nav`: Start navigation mode
- `cap`: Start scene captioning mode
- `sign`: Start road sign detection mode
- `curr`: Start currency detection mode
- `voice`: Toggle voice command recognition
- `speech`: Toggle speech output
- `exit`: Exit the application

### Keyboard Shortcuts (When in a Mode)
- `q`: Return to main menu
- `n`: Switch to navigation mode
- `c`: Switch to captioning mode
- `s`: Switch to sign detection mode
- `m`: Switch to currency detection mode
- `v`: Toggle voice commands

## Project Structure

```
AI_Assistant_for_Blind_People/
├── models/                  # Pre-trained model files
│   ├── yolov8m.pt
│   ├── midas_small.onnx
│   ├── best.pt
│   ├── custom_cnn_model.h5
│   └── vosk-model-small-en-us-0.15/
├── src/
│   ├── main.py              # Main application entry point
│   ├── modes/
│   │   ├── navigation.py    # Navigation assistance functionality
│   │   ├── captioning.py    # Scene description functionality
│   │   ├── sign_detection.py # Road sign detection
│   │   └── currency_detection.py # Currency recognition
│   ├── recognition/
│   │   └── voice_commands.py # Voice command processing
│   ├── tts/
│   │   └── speech_engine.py  # Text-to-speech functionality
│   └── utils/
│       ├── helpers.py       # Helper functions
│       └── ui.py            # User interface utilities
└── requirements.txt         # Required Python packages
```

## Technologies Used

- **YOLOv8**: For object detection
- **MiDaS**: For depth estimation
- **Vosk**: For speech recognition
- **OpenCV**: For computer vision tasks
- **TensorFlow/Keras**: For deep learning models
- **pyttsx3**: For text-to-speech conversion

## Future Improvements

- OCR (Optical Character Recognition) for reading text
- Indoor navigation capabilities
- Additional language support
- Mobile application version
- Integration with GPS for outdoor navigation
- Support for different types of currency

## Contributors

- [Sourabh Patil](https://github.com/sourabh-patil-7)
- [Other Contributors]

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- YOLOv8 by Ultralytics
- MiDaS depth estimation model
- Vosk speech recognition toolkit
- All the open-source libraries that made this project possible