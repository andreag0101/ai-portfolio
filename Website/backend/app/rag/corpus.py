"""Static metadata for the RAG corpus: the technical write-up PDFs already
bundled under frontend/public/deep-learning/reports/ for the Deep Learning
project pages. Shared by the offline index builder (gen_rag_index.py) and
the runtime retrieval/citation code.
"""
from __future__ import annotations

# Each PDF paired with the deep-learning project page it belongs to (see
# frontend/src/lib/deepLearningProjects.ts) so citations can link back to
# both the source PDF (at a specific page) and the write-up's own page.
DOCS = [
    {"file": "hw01-python-oop.pdf", "title": "Python Foundations: OOP & Iterator Protocol", "project_slug": "fundamentals"},
    {"file": "hw02-datasets-dataloaders.pdf", "title": "PyTorch Datasets & DataLoaders", "project_slug": "fundamentals"},
    {"file": "hw03-backprop-optimizers.pdf", "title": "Backpropagation & Optimizers, From Scratch", "project_slug": "backprop-optimizers"},
    {"file": "hw04-cnn-dataset-design.pdf", "title": "Custom CNN Classifier & Dataset Design", "project_slug": "cnn-dataset-design"},
    {"file": "hw05-cnn-depth.pdf", "title": "CNN Architecture Ablations: Depth", "project_slug": "cnn-architecture-ablations"},
    {"file": "hw06-skip-connections.pdf", "title": "CNN Architecture Ablations: Skip Connections", "project_slug": "cnn-architecture-ablations"},
    {"file": "hw07-object-detection.pdf", "title": "Object Detection From Scratch on COCO", "project_slug": "object-detection"},
    {"file": "hw08-semantic-segmentation.pdf", "title": "Semantic Segmentation: mUNet + ASPP", "project_slug": "semantic-segmentation"},
    {"file": "hw09-gan-diffusion.pdf", "title": "GANs vs. Diffusion: Face Generation", "project_slug": "gan-diffusion"},
    {"file": "hw10-transformer-translation.pdf", "title": "Transformer Machine Translation: Post-LN vs. Pre-LN", "project_slug": "transformer-translation"},
]

DOCS_BY_FILE = {d["file"]: d for d in DOCS}

REPORTS_URL_PREFIX = "/deep-learning/reports"
PROJECT_URL_PREFIX = "/deep-learning"


def pdf_href(file: str, page: int) -> str:
    return f"{REPORTS_URL_PREFIX}/{file}#page={page}"


def project_href(project_slug: str) -> str:
    return f"{PROJECT_URL_PREFIX}/{project_slug}"
