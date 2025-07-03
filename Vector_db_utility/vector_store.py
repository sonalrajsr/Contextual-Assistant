import faiss
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_huggingface import HuggingFaceEmbeddings
import uuid
import os


def store_documents_in_faiss(chunks, document_name):
    """
    Store documents in a FAISS vector store.
    Args:
        chunks (list): List of document chunks to be stored.
    """
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    index = faiss.IndexFlatL2(len(embedding_model.embed_query("hello world")))

    vector_store = FAISS(
        embedding_function=embedding_model,
        index=index,
        docstore= InMemoryDocstore(),
        index_to_docstore_id={}
    )

    ids = [str(uuid.uuid4()) for _ in chunks]
    vector_store.add_documents(documents=chunks, ids=ids)
    vector_store.save_local(os.path.join('Data/vector_db', document_name))