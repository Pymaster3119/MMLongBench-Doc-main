import run_vlm
import run_docling_layout
import requests
from PIL import Image
from transformers import AutoConfig
import time

sample_url = "https://huggingface.co/yifeihu/TF-ID   -base/resolve/main/arxiv_2305_10853_5.png?download=true"
sample_image = Image.open(requests.get(sample_url, stream=True).raw).convert("RGB")
print(f"Original image size: {sample_image.size}")

# Get detections (run_docling_layout returns just results, not a tuple)
detections = run_docling_layout.run_docling_layout(sample_image, show=False)

# Get label mapping without loading full model weights
id2label = AutoConfig.from_pretrained("HuggingPanda/docling-layout").id2label


def find_subfigure_boxes_via_layout(pil_img, id2label_map, threshold=0.15, padding=4):
    results = run_docling_layout.run_docling_layout(pil_img, threshold=threshold, show=False)
    if not results:
        return []

    result = results[0]
    boxes = result.get("boxes")
    labels = result.get("labels")
    scores = result.get("scores")
    if boxes is None or labels is None:
        return []

    w, h = pil_img.size
    picture_boxes = []
    for box, label, score in zip(boxes.tolist(), labels.tolist(), scores.tolist()):
        label_name = id2label_map.get(int(label) + 1, str(label))
        if label_name == "Picture" and score >= threshold:
            x1, y1, x2, y2 = map(int, box)
            x1, y1 = max(0, x1 - padding), max(0, y1 - padding)
            x2, y2 = min(w, x2 + padding), min(h, y2 + padding)
            picture_boxes.append((x1, y1, x2, y2))

    if len(picture_boxes) <= 1:
        return []

    # Drop a near-full-image box if present
    picture_boxes = [
        b for b in picture_boxes
        if (b[2] - b[0]) * (b[3] - b[1]) < 0.95 * (w * h)
    ]
    if len(picture_boxes) <= 1:
        return []

    return sorted(picture_boxes, key=lambda b: (b[1], b[0]))


# Process images and tables with VLM
start = time.time()
for result in detections:
    boxes = result["boxes"].tolist()
    labels = result["labels"].tolist()
    for box, label in zip(boxes, labels):
        label_name = id2label.get(int(label)+1, str(label))
        if label_name in ("Picture", "Table"):
            x1, y1, x2, y2 = map(int, box)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(sample_image.width, x2), min(sample_image.height, y2)
            print(f"Cropping {label_name}: ({x1}, {y1}, {x2}, {y2}), size: {x2-x1}x{y2-y1}")
            region_img = sample_image.crop((x1, y1, x2, y2)).convert("RGB")
            print(f"Cropped image size: {region_img.size}")
            
            # Use appropriate prompt based on region type
            if label_name == "Picture":
                sub_boxes = find_subfigure_boxes_via_layout(region_img, id2label)
                if sub_boxes:
                    summaries = []
                    for i, (sx1, sy1, sx2, sy2) in enumerate(sub_boxes, start=1):
                        sub_img = region_img.crop((sx1, sy1, sx2, sy2)).convert("RGB")
                        prompt = (
                            f"Describe subfigure {i} of {len(sub_boxes)} in detail. "
                            "Focus on visual elements, labels, and any embedded text."
                        )
                        summaries.append(run_vlm.run_prompt(sub_img, prompt))
                    print(f"{label_name} Subfigure Summaries:", summaries)
                    print("-" * 80)
                    continue
                else:
                    prompt = "Describe this image in detail. Describe visual elements, labels, and any embedded text. If the image is not recognizable, describe its layout and any visible patterns, such as contours, colors, and textures."
            elif label_name == "Table":
                prompt = "Convert this table into a structured format, listing headers and rows clearly."
            
            summary = run_vlm.run_prompt(region_img, prompt)
            print(f"{label_name} Summary:", summary)
            print("-" * 80)
end = time.time()
print(f"Total processing time: {end - start:.2f} seconds")