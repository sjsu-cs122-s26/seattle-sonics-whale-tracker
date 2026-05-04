# Fish classifier — local training

Train a small transfer-learning classifier on `data/FishImgDataset/{train,val,test}` using the notebook in this folder.

## Environment

**Avoid Homebrew `python@3.14` + global `pip3` for this project.** PyTorch wheels and tooling are still aimed at 3.10–3.12, and some 3.14 installs hit a broken `pyexpat` / `pip` stack (import fails inside `pip` with `Symbol not found: _XML_SetAllocTrackerActivationThreshold` on `pyexpat`). That is a broken interpreter on disk, not this repo’s `requirements-train.txt`.

**Recommended (Apple Silicon / Homebrew):** install a supported Python and use **only** that binary for venv + pip:

```bash
brew install python@3.12
/opt/homebrew/opt/python@3.12/bin/python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -r fish-classifier/requirements-train.txt
python -m ipykernel install --user --name=fish-train --display-name="Fish train"
```

From the **repository root** (if you already have Python 3.11 or 3.12 on `PATH`):

```bash
python3.12 -m venv .venv   # or: python3.11 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
python -m pip install -r fish-classifier/requirements-train.txt
python -m ipykernel install --user --name=fish-train --display-name="Fish train"
```

Always prefer **`python -m pip`** inside the venv so you are not calling a different `pip3` on your PATH.

If you insist on fixing **global** Homebrew 3.14: `brew reinstall python@3.14` (and ensure no conflicting `PYTHONHOME`). A dedicated venv with 3.12 is still simpler.

**NVIDIA GPU (Linux/Windows):** use the CUDA wheel index so PyTorch has CUDA support (same package versions):

```bash
pip install -r fish-classifier/requirements-train.txt --index-url https://download.pytorch.org/whl/cu121
```

## Data

Place the Kaggle fish dataset under `data/FishImgDataset/` with `train/`, `val/`, and `test/` subfolders (class names as subfolder names). This path is gitignored at the repo root.

## Notebook

Open `fish-classifier/notebooks/train_fish_classifier.ipynb` in Jupyter or Cursor, select the `fish-train` kernel (or any interpreter where you installed the requirements), then **Run All**.

- **Mac:** uses MPS when available (`cuda` → `mps` → `cpu`). `PYTORCH_ENABLE_MPS_FALLBACK=1` is set in the notebook for unsupported ops.
- **Teammates without a Mac:** same notebook; CPU or CUDA is chosen automatically. Checkpoints are saved on CPU so they reload on any device.

## Outputs (gitignored)

After a full run, `fish-classifier/models/` contains:

- `fish_model.pt` — weights (`state_dict` on CPU)
- `class_names.json` — label order matching the model output index
- `training_config.json` — image size, normalization, backbone name

For a quick smoke test, set `EPOCHS_HEAD = 1` and `EPOCHS_FT = 0` in the config cell.

## MPS vs CPU/CUDA

Small numerical differences between Metal (MPS), CUDA, and CPU are normal; test accuracy should still agree within about 1–2%.
