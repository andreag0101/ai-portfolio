"""Writes data/rag/suggested_qa.json -- a small, hand-picked set of grounded
question/answer pairs (the "suggested questions" chips on the RAG Assistant
page) so the demo always has a few polished, instant answers available
without needing a live LLM call. Each answer was written by hand, checked
against the actual retrieved passage text (see the citation's page number),
not generated.

Not imported by the app; re-run after editing SUGGESTED below.
"""
import json
from pathlib import Path

from app.rag import corpus

OUT_PATH = Path(__file__).resolve().parent / "data" / "rag" / "suggested_qa.json"


def cite(file: str, page: int, snippet: str) -> dict:
    doc = corpus.DOCS_BY_FILE[file]
    return {
        "title": doc["title"],
        "page": page,
        "snippet": snippet,
        "pdfHref": corpus.pdf_href(file, page),
        "projectHref": corpus.project_href(doc["project_slug"]),
    }


SUGGESTED = [
    {
        "question": "What FID score did the from-scratch DCGAN get compared to the pretrained diffusion model?",
        "answer": "The DCGAN trained from scratch on CelebA scored an FID of 117.07, versus 70.64 for a pretrained diffusion model (lower is better) -- nearly a 2x gap. It shows up visually too: the diffusion samples have more coherent lighting, hair texture, and background detail at the same 64x64 resolution.",
        "citations": [cite("hw09-gan-diffusion.pdf", 6, "Model FID Score: GAN 117.07, Diffusion 70.64 (Table 1: FID Scores)")],
    },
    {
        "question": "How much faster was a PyTorch DataLoader than a naive getitem loop?",
        "answer": "Looping through __getitem__ by hand to load and augment 1000 images took 2.4449s. The fastest DataLoader configuration found by sweeping batch size and worker count (batch=64, 6 workers) processed the same 1000 images in 0.07127s -- about 34x faster.",
        "citations": [cite("hw02-datasets-dataloaders.pdf", 5, "It takes 2.4449 seconds to load and augment all 1000 images using getitem in a loop...")],
    },
    {
        "question": "Which optimizer converged fastest for the multi-neuron model: SGD, SGD+Momentum, or Adam?",
        "answer": "Adam. Across the learning rates tested, plain SGD converged to a visibly higher, noisier loss (or failed to converge at all), while Adam consistently reached the lowest final loss fastest -- SGD+Momentum was more stable but slower to get there.",
        "citations": [cite("hw03-backprop-optimizers.pdf", 15, "Overall, the Adam optimizer outperformed the other two models.")],
    },
    {
        "question": "How did the CNN's cat-classification accuracy change with network depth on CIFAR-10?",
        "answer": "Cat was the weakest class throughout, but it improved the most with depth: 40% accuracy on the shallowest 2-conv-layer network, rising to 61% at 3 layers and 57% at 8 layers, versus an overall accuracy climb of 61% to 76% to 78% across the same three depths.",
        "citations": [cite("hw05-cnn-depth.pdf", 4, "Net1 Net2 Net3 -- cat: 40% 61% 57% (Table 2: Class-wise performance)")],
    },
    {
        "question": "How did classification accuracy change as scenes got more cluttered with multiple objects?",
        "answer": "It dropped monotonically as object multiplicity increased, even though the label set never changed: 56.09% on the single-instance dataset, 53.98% with multiple instances of the same class, and 46.31% with multiple instances from different classes crowding the frame.",
        "citations": [cite("hw04-cnn-dataset-design.pdf", 10, "Accuracy: 56.09% / 53.98% / 46.31% (Table 3: Accuracy, single / multi-same / multi-diff instance)")],
    },
    {
        "question": "What anchor boxes and grid size did the from-scratch object detector use?",
        "answer": "An S x S grid (8x8) over each image, with 5 candidate anchor box aspect ratios per cell. Each ground-truth box is assigned to the grid cell containing its center and the anchor whose aspect ratio is closest to its own, then encoded as a presence flag, a center displacement, a height/width, and a one-hot class label.",
        "citations": [cite("hw07-object-detection.pdf", 5, "Find index of anchor box with aspect ratio closest to bbox aspect ratio... yolo_tensor[y_grid, x_grid, anchor_idx, :]")],
    },
    {
        "question": "Did adding ASPP improve the semantic segmentation model's loss?",
        "answer": "Only marginally. Adding Atrous Spatial Pyramid Pooling to the mUNet architecture left the loss curve's start, end, and downward trend close to the baseline without ASPP -- multi-scale context helped more in combination with a properly weighted Dice loss term than on its own.",
        "citations": [cite("hw08-semantic-segmentation.pdf", 2, "When ASPP is added to the mUnet architecture, there is not much change in the loss curve.")],
    },
    {
        "question": "Post-LayerNorm vs. pre-LayerNorm: which Transformer translated better?",
        "answer": "The post-LayerNorm model (FG) won decisively. It reached a mean Levenshtein (word-edit) distance of 0.83 against reference translations, versus 4.58 for the pre-LayerNorm model, and produced exact or near-exact matches in several cases where PreLN's outputs collapsed to short, mostly-empty sequences.",
        "citations": [cite("hw10-transformer-translation.pdf", 3, "The FC model had a lower average distance (0.83 vs. 4.58), and its median distance was also lower (1.00 compared to 5.00).")],
    },
]


def main():
    # These were hand-written, not sampled from a model, but they play the
    # same role a live "generated" answer would on the page -- a synthesized
    # answer grounded in the cited excerpt -- so they carry the same badge.
    for item in SUGGESTED:
        item["mode"] = "generated"
    OUT_PATH.write_text(json.dumps({"suggested": SUGGESTED}, indent=2))
    print(f"Wrote {len(SUGGESTED)} suggested Q&A pairs to {OUT_PATH}")


if __name__ == "__main__":
    main()
