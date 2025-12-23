import torch
from transformers import pipeline, AutoModelForCausalLM, AutoProcessor, BitsAndBytesConfig, AutoTokenizer
from PIL import Image
import requests

model_id = "microsoft/Phi-3.5-mini-instruct"
device = "cuda"

# ---- bitsandbytes 8-bit config ----
bnb_config = BitsAndBytesConfig(
    load_in_8bit=True,
    llm_int8_threshold=6.0,
    llm_int8_has_fp16_weight=False
)

# ---- load model and processor manually (necessary for quantization) ----
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    device_map="cuda"
)

processor = AutoProcessor.from_pretrained(model_id)

# ---- Create pipeline manually ----
# Explicitly load a tokenizer to avoid inference errors in the pipeline
tokenizer = AutoTokenizer.from_pretrained(model_id)

pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
)


def run_prompt(prompt):
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt}
            ]
        }
    ]

    output = pipe(
        text=messages,
        max_new_tokens=5000,
        temperature=0.1,
        top_p=0.9
    )

    return output[0]["generated_text"][-1]["content"]