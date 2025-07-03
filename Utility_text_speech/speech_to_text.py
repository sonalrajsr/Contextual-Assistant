import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

def speech_to_text(audio_file_path):
    """
    Convert speech to text using a pre-trained Whisper model.
    Args:
        audio_file (str): Path to the audio file to be transcribed.
    Returns:
        str: Transcribed text from the audio file.
    """
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    model_id = "openai/whisper-large-v3-turbo"

    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
    )
    model.to(device)

    processor = AutoProcessor.from_pretrained(model_id)

    pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        torch_dtype=torch_dtype,
        device=device,
    )
    encoded_text = pipe(audio_file_path)
    return encoded_text["text"]