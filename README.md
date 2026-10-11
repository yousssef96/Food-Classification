# FoodVision Mini 🍕🥩🍣

An image classifier that tells whether a food photo is **pizza, steak or sushi**. It uses a pretrained EfficientNetB2 as a feature extractor, tracks experiments with MLflow, and ships as a small Streamlit demo.


## Live Testing: 
https://food-classification-u93rsmdjpfiirngujaijug.streamlit.app/

## Overview

| | |
|---|---|
| **Task** | Image classification (3 classes) |
| **Model** | EfficientNetB2, pretrained on ImageNet, new 3-class classifier head |
| **Data** | `pizza_steak_sushi_20_percent`, a small subset of Food101 |
| **Parameters** | 7,705,221 |
| **Model size** | ~29 MB |
| **Test accuracy** | _94%_ (see [Results](#results)) |
| **Tracking** | MLflow (SQLite backend) |
| **Demo** | Streamlit |

## Project structure

```
.
├── src/
│   ├── app.py          # Streamlit demo
│   └── model.py        # create_effnetb2_model(): model + transforms
├── models/
│   ├── *.pth           # trained weights (state_dict)
│   └── metadata.json   # class names, metrics, params, run ID
├── examples/           # sample images for the demo
├── notebooks/          # training and experiments
├── requirements.txt
└── README.md
```

## Setup

The project uses [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/yousssef96/Food-Classification.git
cd Food-Classification

uv venv
uv pip install -r requirements.txt
```

> If your `requirements.txt` came from `uv pip freeze`, it may pin CUDA builds such as `torch==2.7.1+cu118`. For a CPU-only machine or a cloud deployment, replace them with plain `torch` and `torchvision`.

## Run the demo

```bash
uv run streamlit run src/app.py
```

Then open <http://localhost:8501>. Upload a food photo or pick one of the examples, and the app shows the predicted class, the probabilities for all three classes, and the prediction time.

## Training

Training is done in the notebooks and follows this setup:

- **Transfer learning:** pretrained EfficientNetB2 with its classifier replaced by `Linear(1408, 3)`.
- **Splits:** train / validation / test. The best epoch is chosen on **validation loss**, and the test set is evaluated once at the end.
- **Early stopping** on validation loss, with `ReduceLROnPlateau` lowering the learning rate when it stalls.
- **Reproducibility:** fixed seed (42), and parameters, metrics and the checkpoint are logged to MLflow.

### Experiment tracking

Runs are stored in a local SQLite database. To browse them:

```bash
mlflow ui --backend-store-uri "sqlite:///<path-to>/mlruns.db"
```

Then open <http://127.0.0.1:5000>. Each run logs hyperparameters, per-epoch train/validation metrics, the best epoch, the final test metrics and the model.

## Results

| Model | Test accuracy | Test loss | Parameters | Size |
|---|---|---|---|---|
| EfficientNetB2 | _94%_ | _0.26_ | 7.7M | ~29 MB |

_Update this table with the numbers from your final validation-based run. Add rows here if you compare other models (for example EfficientNetB0 or ViT)._

The test set is small, so differences of a point or two between runs are within noise.

## Model metadata

`models/metadata.json` stores everything needed to use and reproduce the model: class names, architecture, dataset, framework version, seed, final metrics, parameter count, file size and the MLflow run ID. The Streamlit app reads class names from this file.

## Limitations

- The model only knows **three classes**. Any other food (or non-food image) will still be assigned to pizza, steak or sushi, often with high confidence.
- It was trained on a small dataset, so it can struggle with unusual angles, heavy lighting changes, or dishes that look similar to another class.
- The reported accuracy comes from a small test set and should be read as an approximate figure.

## Possible improvements

- Add more food classes and a larger training set.
- Two-phase fine-tuning (train the head first, then unfreeze the backbone at a low learning rate).
- Compare against other architectures and report a per-class confusion matrix.
- Add a confidence threshold with an "unknown food" fallback.
- Deploy the demo on Streamlit Community Cloud or Hugging Face Spaces.

## Tech stack

Python · PyTorch · torchvision · MLflow · Streamlit · scikit-learn · uv



