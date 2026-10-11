Project Structure

```shell
.
├── README.md
├── app
│   ├── app.py
│   └── img
├── assets
├── dataset
│   ├── test
│   ├── train
│   └── val
├── environment.yml
└── src
    ├── 01_data_preparation.ipynb
    ├── 02_ensemble_models.ipynb
    ├── 02_model_inception_resnet_v2.ipynb
    ├── 02_model_resnet50.ipynb
    ├── 02_model_xception.ipynb
    ├── 03_best_model_on_test_xception.ipynb
    └── models
        └── model_Xception_ft.hdf5




### 1. Install Python 3.10

```powershell
winget install Python.Python.3.10
```

### 2. Open PowerShell and go to the project folder

Use your own path (where you extracted the project):

```powershell
cd "C:\path\to\knee_OA_dl_app-main"
```

### 3. Create and activate a virtual environment

```powershell
py -3.10 -m venv .venv310
.\.venv310\Scripts\Activate.ps1
```

If you see `running scripts is disabled on this system`, run this once and activate again:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv310\Scripts\Activate.ps1
```

### 4. Install the libraries

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

This takes a few minutes (TensorFlow is large).

### 5. Run the app

```powershell
python -m streamlit run app\app.py
```

Open **http://localhost:8501** in your browser, upload a knee X-ray, and click
**"Predict Arthrosis in the Knee"**.

- First run takes **20-40 seconds** while the model loads.
- Stop the app with **`Ctrl + C`**.
- Leave the environment with **`deactivate`**.

> Important: always run the commands **from the project root** (the folder that
> contains `app` and `src`), because the app uses relative paths.

---

## The dataset

Download (if not already included):
[Knee Osteoarthritis Dataset with Severity Grading](https://www.kaggle.com/datasets/shashwatwork/knee-osteoarthritis-dataset-with-severity)

The images are already split into three folders. Each has 5 sub-folders named
`0`-`4`, which are the severity grades.

| Folder | Used for | Images |
|--------|----------|--------|
| `dataset/train` | Teaching the model (weights are updated here) | 5,780 |
| `dataset/val` | Checking the model while training | 1,238 |
| `dataset/test` | Final score on unseen images | 1,242 |
| | **Total** | **8,260** |

```
dataset/
├── train/   0 1 2 3 4
├── val/     0 1 2 3 4
└── test/    0 1 2 3 4
```

> The app itself does **not** need the dataset to run. The dataset is only needed
> to **re-train** the model (using the notebooks in `src/`).

---

## How it works

1. **Training** (in `src/02_model_*.ipynb`): a pre-trained Xception network is
   fine-tuned on `dataset/train`, validated on `dataset/val`, then saved as
   `src/models/model_Xception_ft.hdf5`.
2. **Testing** (`src/03_best_model_on_test_xception.ipynb`): the saved model is
   scored on `dataset/test`.
3. **The app** (`app/app.py`): loads the saved model, preprocesses your uploaded
   X-ray to 224x224, predicts the grade, and draws the probability chart + Grad-CAM
   heatmap. A **Download PDF Report** button then exports everything (input image,
   grade, confidence, probabilities table + chart, and Grad-CAM) as a PDF.

If the model file is missing, the app now **downloads it automatically** on
first run (~239 MB, from Google Drive), so you do not need to do this manually.

---

## Deploy to Streamlit Cloud (free)

1. Push this repository to GitHub (dataset and code are already tracked; the model is too large for GitHub and is downloaded by the app itself).
2. Go to **https://share.streamlit.io** and sign in with GitHub.
3. **Create app** -> select `akshau094/Kee-ml`.
4. Set **Branch: `main`** and **Main file path: `app/app.py`**.
5. Click **Deploy**.
   - `requirements.txt` at the repo root is detected automatically.
   - First load takes 20-40 s (the model downloads itself if missing).

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `running scripts is disabled` | Run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass`, then activate again. |
| `ModuleNotFoundError` (tensorflow/streamlit) | The environment is not active. Run `.\.venv310\Scripts\Activate.ps1` and re-install (step 4). |
| Model fails to load | Use Python 3.10 + TensorFlow 2.10 (`.venv310`), not Python 3.12. |
| `FileNotFoundError` for `app/img/mdc.png` | Run the app from the project root (step 5). |
| `Port 8501 is already in use` | Use another port: `python -m streamlit run app\app.py --server.port 8502`. |
| Predictions are all ~20% (random) | The model file is a placeholder. Download the real one (see above). |

---

## Credits

- Original project by **Fernanda Rodriguez (mafda)** — the final project of the
  Complex Data Mining course (MDC013) at Unicamp.
- Original repository: https://github.com/mafda/knee_OA_dl_app
- Dataset: https://www.kaggle.com/datasets/shashwatwork/knee-osteoarthritis-dataset-with-severity
- Techniques: Xception, ResNet50, Inception-ResNet-v2, Grad-CAM.
