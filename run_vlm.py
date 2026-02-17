import torch
from transformers import AutoProcessor
from PIL import Image
import requests
import time

try:
    from transformers import AutoModelForImageTextToText
    _AUTO_MODEL_CLASS = AutoModelForImageTextToText
except Exception:
    from transformers import AutoModelForVision2Seq
    _AUTO_MODEL_CLASS = AutoModelForVision2Seq

# Device setup
device = "cuda" if torch.cuda.is_available() else "cpu"

# Model & processor
model_id = "HuggingFaceTB/SmolVLM2-256M-Video-Instruct"
_model = None
_processor = None


def get_model_and_processor():
    global _model, _processor
    if _model is None or _processor is None:
        _processor = AutoProcessor.from_pretrained(model_id)
        _model = _AUTO_MODEL_CLASS.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
        )
        _model.to(device)
        _model.eval()
    return _model, _processor

# Example image
img_url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/bee.jpg"
img = Image.open(requests.get(img_url, stream=True).raw).convert("RGB")

# Function to run prompt
def run_prompt(image, prompt="Describe this image in detail."):
    model, processor = get_model_and_processor()
    # Build messages format for chat model
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": prompt}
            ]
        }
    ]
    
    # Apply chat template and process
    prompt_text = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = processor(text=prompt_text, images=[image], return_tensors="pt").to(device)
    
    # Generate response
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=False
        )
    
    # Decode only the generated part (skip input tokens)
    generated_ids = output_ids[:, inputs.input_ids.shape[1]:]
    return processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

# Main
if __name__ == "__main__":
    start_time = time.time()
    caption = run_prompt(img)
    print("Output:", caption)
    print("Time taken:", time.time() - start_time, "seconds")