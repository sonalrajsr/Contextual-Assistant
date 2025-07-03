from groq import Groq
import base64
import os

# Function to encode the image
def encode_image(image_path):
  with open(image_path, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode('utf-8')

def image_identification(image_path: str):
    """
    Identify the content of an image using Groq's chat completion API.
    Args:
        image_path (str): The path to the image file.
    Returns:
        str: The response from the Groq API.
    """
    # Getting the base64 string
    base64_image = encode_image(image_path)

    client = Groq(api_key=os.environ.get("GORQ_API"))

    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "You are a medical expert. Carefully analyze the following medical image and provide a detailed diagnosis. What disease or condition is shown in this image??"},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}"
                        }
                    }
                ]
            }
        ],
        model="meta-llama/llama-4-scout-17b-16e-instruct",
    )
    return chat_completion.choices[0].message.content