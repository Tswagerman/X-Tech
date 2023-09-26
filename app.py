import openai
import pyaudio
import wave
import numpy as np
import whisper
import difflib  # Used for string similarity comparison

from flask import Flask, render_template, redirect, url_for

app = Flask(__name__)

# Initialize speech recognition recognizer

# Function to record audio from the microphone
def record_audio():
    audio = pyaudio.PyAudio()
    stream = audio.open(format=FORMAT,
                        channels=CHANNELS,
                        rate=RATE,
                        input=True,
                        frames_per_buffer=CHUNK)
    
    print("Recording...")

    frames = []
    for _ in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
        data = stream.read(CHUNK)
        frames.append(data)

    print("Recording finished.")

    stream.stop_stream()
    stream.close()
    audio.terminate()

    # Save the recorded audio frames as a WAV file  
    with wave.open("audio.wav", "wb") as wav_file:
        wav_file.setnchannels(CHANNELS)
        wav_file.setsampwidth(audio.get_sample_size(FORMAT))
        wav_file.setframerate(RATE)
        wav_file.writeframes(b"".join(frames))

def analyze_audio():
    try:
        model = whisper.load_model("base")
        result = model.transcribe("audio.wav", fp16=False)
        print(result["text"])
        return result["text"]
    except Exception as e:
        print(f"Error analyzing audio: {e}")
        return []

# Function to compare spoken input with desired input using string similarity
def compare_input(spoken_text, desired_text):
    # Calculate a similarity score between the spoken and desired text
    print(f"Spoken text: {spoken_text}")
    similarity = difflib.SequenceMatcher(None, spoken_text, desired_text).ratio()
    return similarity

@app.route('/test')
def test():
    # Initialize variables for desired name and CPR number
    desired_name = "John Doe"  # Replace with the desired name
    desired_cpr = "123456-7890"  # Replace with the desired CPR number

    # Start recording audio
    record_audio()

    # Use speech recognition to transcribe spoken input
    spoken_text = analyze_audio()

    # Compare spoken input to desired input
    name_similarity = compare_input(spoken_text, desired_name)
    cpr_similarity = compare_input(spoken_text, desired_cpr)

    # Determine if the spoken input matches the desired input
    if name_similarity >= 0.8:
        name_result = "Name: Matched"
    else:
        name_result = "Name: Not Matched"

    if cpr_similarity >= 0.8:
        cpr_result = "CPR: Matched"
    else:
        cpr_result = "CPR: Not Matched"

    return f'{name_result}, {cpr_result}'

if __name__ == '__main__':
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 16000
    CHUNK = 1024
    RECORD_SECONDS = 10

    app.run(debug=True)
