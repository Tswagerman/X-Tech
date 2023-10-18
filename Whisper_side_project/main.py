import openai
import pyaudio
import wave
import numpy as np
import whisper



# Set up audio recording parameters
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000  # Adjust the sample rate as needed
CHUNK = 1024
RECORD_SECONDS = 20  # Adjust the duration of each recording as needed

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

# Function to send audio to Whisper for analysis
def analyze_audio():
    try:
        model = whisper.load_model("base")
        result = model.transcribe("audio.wav", fp16=False)
        print(result["text"])
        return result["text"]
    except Exception as e:
        print(f"Error analyzing audio: {e}")
        return []

# Main function for glaucoma detection
def main():
    try:
        while True:
            # Record audio from the microphone
            record_audio()

            # Send audio to Whisper for analysis
            transcriptions = analyze_audio()

            # Process and analyze transcriptions
            
    except KeyboardInterrupt:
        print("Exiting...")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()