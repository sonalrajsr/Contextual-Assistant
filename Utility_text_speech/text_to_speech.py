import os
from groq import Groq

def text_to_speech(text, speech_file_path = "speech.wav", model="playai-tts", voice="Fritz-PlayAI"):
    """
    Converts text to speech using the Groq API.
    """
    client = Groq(api_key=os.environ.get("GORQ_API"))
    response = client.audio.speech.create(
        model=model,
        voice=voice,
        input=text,
        response_format="wav"
    )
    audio_file = response.write_to_file(os.path.join('Data/audio_files', speech_file_path))
    return audio_file