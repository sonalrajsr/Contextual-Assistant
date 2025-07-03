def query_faiss_vector_store(vector_store, query, k=3):
    """
    Query the FAISS vector store for similar documents.
    Args:
        vector_store (FAISS): The FAISS vector store instance.
        query (str): The query string to search for.
        k (int): The number of similar documents to return.
    Returns:
        list: List of similar documents.
    """
    results = vector_store.similarity_search(query=query, k=k)
    return results