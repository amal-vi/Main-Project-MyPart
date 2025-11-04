# server.py
from flask import Flask, render_template
from flask_socketio import SocketIO, emit
from faster_whisper import WhisperModel
import numpy as np
import io
import wave

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key'
socketio = SocketIO(app, cors_allowed_origins="*", max_http_buffer_size=16000000)

# Initialize Faster-Whisper model (use 'base' or 'small' for lower latency)
model = WhisperModel("base", device="cpu", compute_type="int8")
# For GPU: model = WhisperModel("base", device="cuda", compute_type="float16")

audio_buffer = []

@app.route('/')
def index():
    return render_template('index.html')

@socketio.on('audio_chunk')
def handle_audio_chunk(data):
    """Process incoming audio chunks in real-time"""
    try:
        # Convert base64 audio to numpy array
        audio_data = np.frombuffer(data, dtype=np.int16)
        audio_buffer.append(audio_data)
        
        # Process when buffer reaches ~1 second (adjust for latency vs accuracy)
        if len(audio_buffer) >= 2:  # 2 chunks = ~1 second at 16kHz
            # Concatenate buffer
            full_audio = np.concatenate(audio_buffer)
            audio_float = full_audio.astype(np.float32) / 32768.0
            
            # Transcribe with Faster-Whisper
            segments, info = model.transcribe(
                audio_float,
                language="en",
                vad_filter=True,  # Voice Activity Detection for better accuracy
                beam_size=1,      # Faster inference
                best_of=1,        # Faster inference
            )
            
            # Send transcription back to client
            text = " ".join([segment.text for segment in segments])
            if text.strip():
                emit('transcription', {'text': text})
            
            # Clear buffer
            audio_buffer.clear()
            
    except Exception as e:
        print(f"Error: {e}")
        emit('error', {'message': str(e)})

@socketio.on('stop_recording')
def handle_stop():
    """Process any remaining audio when recording stops"""
    if audio_buffer:
        try:
            full_audio = np.concatenate(audio_buffer)
            audio_float = full_audio.astype(np.float32) / 32768.0
            segments, info = model.transcribe(audio_float, language="en", vad_filter=True)
            text = " ".join([segment.text for segment in segments])
            if text.strip():
                emit('transcription', {'text': text})
            audio_buffer.clear()
        except Exception as e:
            print(f"Error: {e}")

if __name__ == '__main__':
    socketio.run(app, debug=True, host='0.0.0.0', port=5001)
