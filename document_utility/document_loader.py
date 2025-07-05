from langchain_community.document_loaders import PyPDFLoader

def documents_loader(path: str):
    """
    Load documents from a given path.
    Args:
        path (str): The path to the document.
    Returns:
        list: A list of loaded documents.
    """
    loader = PyPDFLoader(path)
    documents = loader.load()
    return documents
