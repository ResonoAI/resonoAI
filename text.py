import google.generativeai as genai
import speech_recognition as sr
import pyaudio
import wave
import audioop
import time

genai.configure(api_key="AIzaSyDKjcaRvFsETzC3x7LFp9Xr4e99o5C3igs")
model = genai.GenerativeModel('gemini-pro')

def speech_to_text_converter(audio_file):
    recognizer = sr.Recognizer()
    with sr.AudioFile(audio_file) as source:
        audio_data = recognizer.record(source)
        try:
            transcribed_text = recognizer.recognize_google(audio_data)
            if not transcribed_text.strip():
                return "caller did not say anything"
            else:
                return transcribed_text
        except sr.UnknownValueError:
            return "caller did not say anything"
        except sr.RequestError as e:
            print("Could not request results from Google Speech Recognition service; {0}".format(e))
            return "caller did not say anything"

def summarize_call(transcribed_text):
    summary = transcribed_text + "..."
    return summary

def generate_additional_info(summary):
    return model.generate_content("You are an AI that provides helpful information to a 911 first responder."
                                  " Your response should be formatted like this: Name:, Address:, Condition:, Level of Pain:, Surroundings:, and Additional Information:" 
                                  + "I need you to source the information from this caller:" + summary).text

def record_audio(filename="recorded_speech.wav", max_silence_duration=5):
    CHUNK = 1024
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 44100

    audio = pyaudio.PyAudio()

    stream = audio.open(format=FORMAT, channels=CHANNELS,
                        rate=RATE, input=True,
                        frames_per_buffer=CHUNK)

    print("Recording...")

    frames = []
    start_time = time.time()
    silent = False

    while True:
        data = stream.read(CHUNK)
        frames.append(data)
        rms = audioop.rms(data, 2)
        if rms < 3250:
            if not silent:
                silent_start_time = time.time()
                silent = True
            else:
                if time.time() - silent_start_time >= max_silence_duration:
                    break
        else:
            silent = False
        
    print("Finished recording.")

    stream.stop_stream()
    stream.close()
    audio.terminate()

    wf = wave.open(filename, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(audio.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))
    wf.close()


audio_file = "911_call_audio.wav"
record_audio(audio_file)

transcribed_text = speech_to_text_converter(audio_file)
summary = summarize_call(transcribed_text)
additional_info = generate_additional_info(summary)

print("Summary of the 911 call:")
print(summary)
print("Additional information:")
print(additional_info)
