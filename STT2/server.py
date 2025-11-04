# server.py
from flask import Flask, render_template
from flask_socketio import SocketIO, emit
import speech_recognition as sr
import io
import wave
import threading

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key'
socketio = SocketIO(app, cors_allowed_origins="*", max_http_buffer_size=16000000)

recognizer = sr.Recognizer()
recognizer.energy_threshold = 300  # Adjust for ambient noise
recognizer.dynamic_energy_threshold = True
recognizer.pause_threshold = 0.8  # Seconds of silence to consider phrase complete

@app.route('/')
def index():
    return render_template('index.html')

@socketio.on('audio_data')
def handle_audio(audio_data):
    """Process audio data received from client"""
    try:
        # Convert raw audio bytes to AudioData object
        audio = sr.AudioData(
            audio_data,
            sample_rate=16000,
            sample_width=2  # 16-bit audio = 2 bytes
        )
        
        # Recognize speech using different engines
        # Option 1: Google Speech Recognition (free, requires internet)
        text = recognizer.recognize_google(audio, language='en-US')
        
        # Option 2: Sphinx (offline, install with: pip install pocketsphinx)
        # text = recognizer.recognize_sphinx(audio)
        
        # Option 3: Whisper API (requires OpenAI API key)
        # text = recognizer.recognize_whisper_api(audio, api_key="your-key")
        
        if text:
            emit('transcription', {'text': text})
            
    except sr.UnknownValueError:
        # Speech was unintelligible
        pass
    except sr.RequestError as e:
        emit('error', {'message': f'Service error: {str(e)}'})
    except Exception as e:
        emit('error', {'message': f'Error: {str(e)}'})

@socketio.on('microphone_stream')
def handle_microphone_stream():
    """Alternative: Use server-side microphone (if running locally)"""
    def listen():
        with sr.Microphone(sample_rate=16000) as source:
            # Adjust for ambient noise
            recognizer.adjust_for_ambient_noise(source, duration=1)
            
            while True:
                try:
                    # Listen for audio
                    audio = recognizer.listen(source, timeout=1, phrase_time_limit=5)
                    
                    # Recognize speech
                    text = recognizer.recognize_google(audio, language='en-US')
                    
                    if text:
                        socketio.emit('transcription', {'text': text})
                        
                except sr.WaitTimeoutError:
                    continue
                except sr.UnknownValueError:
                    continue
                except Exception as e:
                    print(f"Error: {e}")
                    break
    
    # Run in separate thread
    thread = threading.Thread(target=listen)
    thread.daemon = True
    thread.start()

if __name__ == '__main__':
    socketio.run(app, debug=True, host='0.0.0.0', port=5002)
