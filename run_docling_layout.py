import torch
from PIL import Image, ImageDraw, ImageFont
from transformers import RTDetrForObjectDetection, RTDetrImageProcessor
import requests
import matplotlib.pyplot as plt


def suppress_contained(detections, margin_pixels: float = 10.0):
    """Remove boxes fully inside larger, higher-score boxes of the same label."""
    filtered = []
    for result in detections:
        boxes = result["boxes"].tolist()
        scores = result["scores"].tolist()
        labels = result["labels"].tolist()

        keep = []
        for i, (b_small, s_small, l_small) in enumerate(zip(boxes, scores, labels)):
            x1s, y1s, x2s, y2s = b_small
            area_small = (x2s - x1s) * (y2s - y1s)
            contained = False
            for j, (b_big, s_big, l_big) in enumerate(zip(boxes, scores, labels)):
                if i == j or l_small != l_big:
                    continue
                x1b, y1b, x2b, y2b = b_big
                if (
                    x1b - margin_pixels <= x1s
                    and y1b - margin_pixels <= y1s
                    and x2b + margin_pixels >= x2s
                    and y2b + margin_pixels >= y2s
                ):
                    area_big = (x2b - x1b) * (y2b - y1b)
                    if area_big > area_small * 1.1:
                        contained = True
                        break
            if not contained:
                keep.append((scores[i], labels[i], result["boxes"][i]))

        if keep:
            kept_scores, kept_labels, kept_boxes = zip(*keep)
            filtered.append({
                "scores": torch.tensor(kept_scores),
                "labels": torch.tensor(kept_labels),
                "boxes": torch.stack(list(kept_boxes)),
            })
        else:
            filtered.append({"scores": torch.tensor([]), "labels": torch.tensor([]), "boxes": torch.empty((0, 4))})

    return filtered

model_id = "HuggingPanda/docling-layout"
image_processor = RTDetrImageProcessor.from_pretrained(model_id)
model = RTDetrForObjectDetection.from_pretrained(model_id)

def run_docling_layout(
    image: Image.Image,
    threshold: float = 0.3,
    margin_pixels: float = 10.0,
    show: bool = True,
):

    resize_cfg = {"height": 640, "width": 640}
    inputs = image_processor(images=image, return_tensors="pt", size=resize_cfg)

    with torch.no_grad():
        outputs = model(**inputs)

    results = image_processor.post_process_object_detection(
        outputs,
        target_sizes=torch.tensor([image.size[::-1]]),
        threshold=threshold,
    )

    results = suppress_contained(results, margin_pixels=margin_pixels)

    return results


if __name__ == "__main__":
    sample_url = "https://huggingface.co/yifeihu/TF-ID-base/resolve/main/arxiv_2305_10853_5.png?download=true"
    sample_image = Image.open(requests.get(sample_url, stream=True).raw)
    run_docling_layout(sample_image)
