import gradio as gr
from gtts import gTTS
import tempfile
from document_utility.document_loader import documents_loader
from document_utility.split_document import split_documents
from Utility_text_speech.image_identification import image_identification, encode_image
from Utility_text_speech.text_to_speech import text_to_speech
from Utility_text_speech.speech_to_text import speech_to_text
from Utility_text_speech.text_to_text import text_to_text
from Vector_db_utility.vector_store import store_documents_in_faiss
from Vector_db_utility.vector_db_query import query_faiss_vector_store

def process_all(query, audio, image, pdf):
    warning = ""
    if audio is not None and query and query.strip():
        warning = "Both text and audio provided. Audio will be used."
        text_from_audio = speech_to_text(audio)
        query = text_from_audio if text_from_audio else "No query provided."
    elif audio is not None:
        text_from_audio = speech_to_text(audio)
        query = text_from_audio if text_from_audio else "No query provided."
    elif query and query.strip():
        query = query.strip()
    else:
        query = "No query provided."

    if image is not None:
        image_result = image_identification(image)
    else:
        image_result = ""

    if pdf is not None:
        document = documents_loader(pdf)
        chunks = split_documents(document)
        vector_store = store_documents_in_faiss(chunks, pdf.name)
        context = query_faiss_vector_store(vector_store, query)
        pdf_result = context if context else "No relevant information found in PDF."
    else:
        pdf_result = ""

    llm_prompt = f"User Query: {query}\nImage Result: {image_result}\nPDF Result: {pdf_result}"
    result = f"{warning}\n{text_to_text(llm_prompt)}" if warning else f"{text_to_text(llm_prompt)}"

    # Generate TTS audio for result
    tts = gTTS(result)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as fp:
        tts.save(fp.name)
        audio_path = fp.name

    return result, audio_path

with gr.Blocks() as demo:
    gr.Markdown("# Medical Super Bot")

    with gr.Row():
        with gr.Column():
            gr.Markdown("**Type your question OR record your question (not both):**")
            query = gr.Textbox(label="Type your question")
            audio = gr.Audio(sources="microphone", type="filepath", label="Or record your question")
            image = gr.Image(type="filepath", label="Upload an image")
            pdf = gr.File(label="Upload a PDF/Book", file_types=[".pdf"])
            submit = gr.Button("Submit")
        with gr.Column():
            result = gr.Textbox(label="Result", interactive=False)
            tts_audio = gr.Audio(label="🔊 Listen to Result")

    submit.click(
        process_all,
        inputs=[query, audio, image, pdf],
        outputs=[result, tts_audio]
    )

demo.launch()