import openai
import pyaudio
import numpy as np

# Initialize OpenAI API with your API key
openai.api_key = "YOUR_API_KEY"

# Set up audio recording parameters
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000  # Adjust the sample rate as needed
CHUNK = 1024
RECORD_SECONDS = 5  # Adjust the duration of each recording as needed

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

    return b''.join(frames)

# Function to send audio to Whisper for analysis
def analyze_audio(audio_data):
    try:
        response = openai.Audio.create(
            model="whisper-large",
            audio=audio_data,
            # Other optional parameters like max_tokens, temperature, etc.
        )
        return response['transcriptions']
    except Exception as e:
        print(f"Error analyzing audio: {e}")
        return []

# Main function for glaucoma detection
def main():
    try:
        while True:
            # Record audio from the microphone
            audio_data = record_audio()

            # Send audio to Whisper for analysis
            transcriptions = analyze_audio(audio_data)

            # Process and analyze transcriptions for glaucoma detection
            # Implement your glaucoma detection logic here
            
    except KeyboardInterrupt:
        print("Exiting...")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
