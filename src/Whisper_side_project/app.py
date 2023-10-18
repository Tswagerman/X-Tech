import openai
import pyaudio
import wave
import numpy as np
import whisper
import difflib  # Used for string similarity comparison

from flask import Flask, render_template, redirect, url_for, request, session

app = Flask(__name__)

# Read API key from api.txt file
with open("api.txt", "r") as f:
    api_key = f.read().strip()

openai.api_key = api_key
app.secret_key = api_key

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
        result = model.transcribe("audio.wav", fp16=False, language="da")
        print(result["text"])
        return result["text"]
    except Exception as e:
        print(f"Error analyzing audio: {e}")
        return []
    
def compare_input(spoken_text, desired_text):
    # Calculate a similarity score between the spoken and desired text
    print(f"Spoken text: {spoken_text}")
    similarity = difflib.SequenceMatcher(None, spoken_text, desired_text).ratio()
    return similarity

@app.route('/')
def index():
    session.clear()  # Clear the session data
    return redirect(url_for('name_confirmation'))

@app.route('/test_interface')
def test_interface():
    return render_template('test_interface.html')

@app.route('/name_confirmation', methods=['GET', 'POST'])
def name_confirmation():
    desired_name = "Ja"  # Replace with the desired name

    if request.method == 'POST':
        # Start recording audio
        record_audio()

        # Use speech recognition to transcribe spoken input
        spoken_text = analyze_audio()

        # Compare spoken input to desired input
        name_similarity = compare_input(spoken_text, desired_name)
        print("Name similarity: ", name_similarity)

        # Store the name similarity score in the session
        session['name_similarity'] = name_similarity

        return redirect(url_for('cpr_confirmation'))

    return render_template('name_confirmation.html')

@app.route('/cpr_confirmation', methods=['GET', 'POST'])
def cpr_confirmation():
    desired_cpr = "nej"  # Replace with the desired CPR number

    if request.method == 'POST':
        # Start recording audio
        record_audio()

        # Use speech recognition to transcribe spoken input
        spoken_text = analyze_audio()

        # Compare spoken input to desired input
        cpr_similarity = compare_input(spoken_text, desired_cpr)
        print("CPR similarity: ", cpr_similarity)

        # Store the CPR similarity score in the session
        session['cpr_similarity'] = cpr_similarity

        return redirect(url_for('result'))

    return render_template('cpr_confirmation.html')

@app.route('/result')
def result():
    name_similarity = session.get('name_similarity', 0)
    cpr_similarity = session.get('cpr_similarity', 0)

    name_result = "Name: Matched" if name_similarity >= 0.8 else "Name: Not Matched"
    cpr_result = "CPR: Matched" if cpr_similarity >= 0.8 else "CPR: Not Matched"

    return f'{name_result}, {cpr_result}'

if __name__ == '__main__':
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 16000
    CHUNK = 1024
    RECORD_SECONDS = 2

    app.run(debug=True)