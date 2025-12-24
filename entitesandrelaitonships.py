import run_unichart
import run_docling_layout
import requests
from PIL import Image
from transformers import AutoConfig

sample_url = "https://huggingface.co/yifeihu/TF-ID-base/resolve/main/arxiv_2305_10853_5.png?download=true"
sample_image = Image.open(requests.get(sample_url, stream=True).raw).convert("RGB")
print(f"Original image size: {sample_image.size}")

# Get detections (run_docling_layout returns just results, not a tuple)
detections = run_docling_layout.run_docling_layout(sample_image, show=False)

# Get label mapping without loading full model weights
id2label = AutoConfig.from_pretrained("HuggingPanda/docling-layout").id2label

# Send all images and tables to UniChart and white them out in the original image
for result in detections:
    boxes = result["boxes"].tolist()
    labels = result["labels"].tolist()
    for box, label in zip(boxes, labels):
        label_name = id2label.get(int(label)+1, str(label))
        if label_name == "Picture":
            x1, y1, x2, y2 = map(int, box)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(sample_image.width, x2), min(sample_image.height, y2)
            print(f"Cropping {label_name}: ({x1}, {y1}, {x2}, {y2}), size: {x2-x1}x{y2-y1}")
            chart_img = sample_image.crop((x1, y1, x2, y2)).convert("RGB")
            print(f"Cropped image size: {chart_img.size}")
            summary = run_unichart.process_chart(chart_img)
            print("Chart Summary:", summary)