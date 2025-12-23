import torch
from transformers import pipeline, AutoModelForVision2Seq, AutoProcessor, BitsAndBytesConfig
from PIL import Image
import requests

model_id = "Qwen/Qwen3-VL-2B-Instruct"
device = "cuda"

bnb_config = BitsAndBytesConfig(
    load_in_8bit=True,
    llm_int8_threshold=6.0,
    llm_int8_has_fp16_weight=False
)

model = AutoModelForVision2Seq.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    device_map="cuda"
)

processor = AutoProcessor.from_pretrained(model_id)

pipe = pipeline(
    "image-text-to-text",
    model=model,
    processor=processor,
)

img = Image.open(requests.get(
    "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/bee.jpg",
    stream=True
).raw).convert("RGB")


def run_prompt(img, prompt):
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": img},
                {"type": "text", "text": prompt}
            ]
        }
    ]

    output = pipe(
        text=messages,
        max_new_tokens=500,
        temperature=0.1,
        top_p=0.9
    )

    return output[0]["generated_text"][-1]["content"]