from groq import Groq
import os

def text_to_text(text: str):
    """
    Converts text to text using the Groq API.
    Args:
        text (str): The input text to be processed.
    Returns:
        str: The response from the Groq API.
    """
    client = Groq(api_key=os.environ.get("GORQ_API"))
    completion = client.chat.completions.create(
        model="meta-llama/llama-4-scout-17b-16e-instruct",
        messages=[
        {
            "role": "user",
            "content": text
        }
        ],
        temperature=1,
        max_completion_tokens=1024,
        top_p=1,
        stream=True,
        stop=None,
    )
    response = ""
    for chunk in completion:
        content = chunk.choices[0].delta.content or ""
        response += content

    return response.strip()