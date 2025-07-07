import os

def list_pdfs(folder="Data/PDF_Files"):
    """List all PDF files in the specified folder."""
    all_files = os.listdir(folder)
    return all_files if all_files else ["No PDF files found"]