import torch
from transformers import DonutProcessor, VisionEncoderDecoderModel
from PIL import Image
import requests
import matplotlib.pyplot as plt

# Choose a UniChart model checkpoint
model_name = "ahmed-masry/unichart-base-960"

# Load
processor = DonutProcessor.from_pretrained(model_name)
model = VisionEncoderDecoderModel.from_pretrained(model_name)
device = torch.device("cuda")
model.to(device)


def process_chart(img):
    plt.imshow(img)
    plt.axis("off")
    plt.show()
    # Craft a summarization prompt
    prompt = "<summarize_chart> <s_answer>"

    # Tokenize input: prompt + chart image
    pixel_values = processor(img, return_tensors="pt").pixel_values.to(device)
    decoder_input_ids = processor.tokenizer(prompt, add_special_tokens=False, return_tensors="pt").input_ids.to(device)

    # Generate text
    outputs = model.generate(
        pixel_values=pixel_values,
        decoder_input_ids=decoder_input_ids,
        max_length=model.decoder.config.max_position_embeddings,
        num_beams=4,
        early_stopping=True,
    )

    # Decode (strip special tokens)
    generated = processor.batch_decode(outputs, skip_special_tokens=True)[0]
    result = generated.replace(prompt, "").strip()
    return result

if __name__ == "__main__":
    import time
    chart_url = "https://raw.githubusercontent.com/vis-nlp/ChartQA/main/ChartQA%20Dataset/val/png/multi_col_1229.png"
    img = Image.open(requests.get(chart_url, stream=True).raw).convert("RGB")
    start = time.time()
    summary = process_chart(img)
    end = time.time()
    print("Chart Summary:", summary)
    print("Processing Time:", end - start, "seconds")