import torch
from PIL import Image, ImageDraw, ImageFont
from transformers import RTDetrForObjectDetection, RTDetrImageProcessor
import requests
import matplotlib.pyplot as plt


def suppress_contained(detections, margin_pixels: float = 10.0, prefer_inner: bool = True):
    filtered = []
    for result in detections:
        boxes = result["boxes"].tolist()
        scores = result["scores"].tolist()
        labels = result["labels"].tolist()

        suppress = set()
        for i, (b_i, l_i) in enumerate(zip(boxes, labels)):
            x1i, y1i, x2i, y2i = b_i
            area_i = (x2i - x1i) * (y2i - y1i)
            for j, (b_j, l_j) in enumerate(zip(boxes, labels)):
                if i == j or l_i != l_j:
                    continue
                x1j, y1j, x2j, y2j = b_j
                contains_i_in_j = (
                    x1j - margin_pixels <= x1i
                    and y1j - margin_pixels <= y1i
                    and x2j + margin_pixels >= x2i
                    and y2j + margin_pixels >= y2i
                )
                if not contains_i_in_j:
                    continue
                area_j = (x2j - x1j) * (y2j - y1j)
                if area_j > area_i * 1.1:
                    if prefer_inner:
                        suppress.add(j)
                    else:
                        suppress.add(i)

        keep = [
            (scores[i], labels[i], result["boxes"][i])
            for i in range(len(boxes))
            if i not in suppress
        ]

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
    prefer_inner: bool = True,
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

    results = suppress_contained(results, margin_pixels=margin_pixels, prefer_inner=prefer_inner)

    return results


if __name__ == "__main__":
    sample_url = "https://huggingface.co/yifeihu/TF-ID-base/resolve/main/arxiv_2305_10853_5.png?download=true"
    sample_image = Image.open(requests.get(sample_url, stream=True).raw)
    run_docling_layout(sample_image)
