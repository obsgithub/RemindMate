from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage
import base64
from PIL import Image
import io
import time

# Initialize the ChatOllama model instance
# Note: Users must provide their own deployed model name and host URL.
chat = ChatOllama(
    model="YOUR_MODEL_NAME_HERE",  # e.g., "qwen2.5vl:72b"
    temperature=0,
    base_url="http://YOUR_OLLAMA_HOST_IP:PORT"  # Placeholder for privacy/security
)

def image_to_base64(image_path):
    """
    Reads an image from the local path and converts it to a base64 encoded string.
    """
    with Image.open(image_path) as img:
        buffer = io.BytesIO()
        img.save(buffer, format="PNG")
        img_bytes = buffer.getvalue()
        img_base64 = base64.b64encode(img_bytes).decode("utf-8")
        return img_base64

def analyze_image(img_path, prompt, api_key=None):
    """
    Constructs the multimodal message and invokes the vision model.
    """
    local_image_path = img_path
    image_base64 = image_to_base64(local_image_path)
    
    start_time = time.time()
    message = HumanMessage(
        content=[
            {"type": "text", "text": prompt},
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"},
            },
        ],
    )
    
    response = chat.invoke([message])
    end_time = time.time()
    
    # Optional: Log inference time
    # print(f"Model Inference Time: {end_time - start_time:.4f} seconds")
    
    return response.content

if __name__ == "__main__":
    # Example usage for demonstration purposes only.
    # This block requires valid paths and prompts to execute successfully.
    
    start_time_total = time.time()
    
    # Placeholders for the target image and the query
    sample_image_path = "/path/to/your/dataset/image.png"
    sample_prompt = "Describe the contents of this image." 
    
    # Uncomment the following line to test when configured properly:
    # result = analyze_image(sample_image_path, sample_prompt)
    # print(result)
    
    end_time_total = time.time()
    # print(f"Total Execution Time: {end_time_total - start_time_total:.4f} seconds")