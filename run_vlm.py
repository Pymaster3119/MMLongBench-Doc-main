import torch
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import requests
import time

# Device setup
device = "cuda" if torch.cuda.is_available() else "cpu"

# Model & processor
model_id = "Salesforce/blip-image-captioning-base"
processor = BlipProcessor.from_pretrained(model_id)
model = BlipForConditionalGeneration.from_pretrained(model_id)
model.to(device)
model.eval()

# Example image
img_url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/bee.jpg"
img = Image.open(requests.get(img_url, stream=True).raw).convert("RGB")

# Function to run prompt
def run_prompt(image, prompt=None):
    # Preprocess image (and optional prompt)
    inputs = processor(images=image, return_tensors="pt").to(device)

    # Generate caption
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            do_sample=False
        )

    return processor.decode(output_ids[0], skip_special_tokens=True)

# Main
if __name__ == "__main__":
    start_time = time.time()
    caption = run_prompt(img)
    print("Output:", caption)
    print("Time taken:", time.time() - start_time, "seconds")