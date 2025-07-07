import os
import gradio as gr
from gtts import gTTS
import tempfile
import shutil
from document_utility.document_loader import documents_loader
from document_utility.split_document import split_documents
from document_utility.all_pdfs import list_pdfs
from Utility_text_speech.image_identification import image_identification, encode_image
from Utility_text_speech.text_to_speech import text_to_speech
from Utility_text_speech.speech_to_text import speech_to_text
from Utility_text_speech.text_to_text import text_to_text
from Vector_db_utility.vector_store import store_documents_in_faiss, load_vector_store
from Vector_db_utility.vector_db_query import query_faiss_vector_store

DATA_FOLDER = "Data/PDF_Files"
os.makedirs(DATA_FOLDER, exist_ok=True)
def get_kb_choices(path = "Data/vector_db"):
    pdfs = list_pdfs(path)
    return ["None"] + pdfs if pdfs else ["None"]

def process_all(query, audio, image, pdf, kb_pdf):
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

    # If user uploads a PDF, use it and ignore kb_pdf
    if pdf is not None:
        pdf_name = os.path.basename(getattr(pdf, "name", os.path.basename(pdf)))
        save_path = os.path.join(DATA_FOLDER, pdf_name)
        shutil.copy(pdf, save_path)
        document = documents_loader(save_path)
        chunks = split_documents(document)
        vector_store = store_documents_in_faiss(chunks, pdf_name)
        context = query_faiss_vector_store(vector_store, query)
        pdf_result = context if context else "No relevant information found in PDF."
    elif kb_pdf and kb_pdf != "None":
        kb_pdf_db_path = os.path.join(kb_pdf + ".faiss")
        vector_store = load_vector_store(kb_pdf_db_path)
        context = query_faiss_vector_store(vector_store, query)
        pdf_result = context if context else "No relevant information found in PDF."
    else:
        pdf_result = "No PDF provided or selected from knowledge base."

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
            kb_pdf = gr.Dropdown(label="Use Knowledge Base PDFs", choices=get_kb_choices(), value="None")
            submit = gr.Button("Submit")
        with gr.Column():
            result = gr.Textbox(label="Result", interactive=False)
            tts_audio = gr.Audio(label="🔊 Listen to Result")

    submit.click(
        process_all,
        inputs=[query, audio, image, pdf, kb_pdf],
        outputs=[result, tts_audio]
    )

demo.launch()