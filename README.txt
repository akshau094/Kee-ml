KNEE OSTEOARTHRITIS DETECTION APP
=================================

A Streamlit web app that predicts knee osteoarthritis severity from an X-ray
image using a trained deep learning model (Xception). Upload a knee X-ray and
it shows the grade, the confidence, and a heatmap of what the model looked at.

Grades (KL score):
  0 = Healthy
  1 = Doubtful
  2 = Minimal
  3 = Moderate
  4 = Severe


PROJECT FILES
-------------

  app/app.py
      Main file - the web app. This is the one you run.

  src/models/model_Xception_ft.hdf5
      The trained model the app loads (~239 MB).

  src/*.ipynb
      Notebooks that trained and evaluated the model.

  dataset/
      8,260 knee X-ray images, split into train, val, test.

  requirements.txt
      The Python libraries to install.

  environment.yml
      Same libraries, for Conda users (optional).

  assets/
      Images used in the docs only (not needed to run).


QUICK START (WINDOWS)
---------------------

1. Install Python 3.10

     winget install Python.Python.3.10

2. Open PowerShell and go to the project folder (use your own path)

     cd "C:\path\to\knee_OA_dl_app-main"

3. Create and activate a virtual environment

     py -3.10 -m venv .venv310
     .\.venv310\Scripts\Activate.ps1

   If you see "running scripts is disabled on this system", run this once and
   activate again:

     Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
     .\.venv310\Scripts\Activate.ps1

4. Install the libraries

     python -m pip install --upgrade pip
     python -m pip install -r requirements.txt

   This takes a few minutes (TensorFlow is large).

5. Run the app

     python -m streamlit run app\app.py

   Open http://localhost:8501 in your browser, upload a knee X-ray, and click
   "Predict Arthrosis in the Knee".

   - First run takes 20-40 seconds while the model loads.
   - Stop the app with Ctrl + C.
   - Leave the environment with: deactivate

   Important: always run the commands from the project root (the folder that
   contains "app" and "src"), because the app uses relative paths.


THE DATASET
-----------

Download (if not already included):
  https://www.kaggle.com/datasets/shashwatwork/knee-osteoarthritis-dataset-with-severity

The images are already split into three folders. Each has 5 sub-folders named
0-4, which are the severity grades.

  Folder           Used for                                    Images
  ---------------  ------------------------------------------  ------
  dataset/train    Teaching the model (weights updated here)    5,780
  dataset/val      Checking the model while training            1,238
  dataset/test     Final score on unseen images                 1,242
                                                              Total: 8,260

  dataset/
    train/   0 1 2 3 4
    val/     0 1 2 3 4
    test/    0 1 2 3 4

  The app itself does not need the dataset to run. The dataset is only needed
  to re-train the model (using the notebooks in src/).


HOW IT WORKS
------------

1. Training (in src/02_model_*.ipynb): a pre-trained Xception network is
   fine-tuned on dataset/train, validated on dataset/val, then saved as
   src/models/model_Xception_ft.hdf5.

2. Testing (src/03_best_model_on_test_xception.ipynb): the saved model is
   scored on dataset/test.

3. The app (app/app.py): loads the saved model, preprocesses your uploaded
   X-ray to 224x224, predicts the grade, and draws the probability chart +
   Grad-CAM heatmap. A "Download PDF Report" button then exports everything
   (input image, grade, confidence, probabilities table + chart, and Grad-CAM)
   as a PDF.

If the model file is missing, download it and put it in src/models/:

  python -m gdown "https://drive.google.com/uc?id=1vPS_j2AW3M1W8GydREEDw6CDENSgVwCy" -O "src\models\model_Xception_ft.hdf5"


TROUBLESHOOTING
---------------

  Problem                               Fix
  ------------------------------------  ---------------------------------------
  running scripts is disabled           Run: Set-ExecutionPolicy -Scope Process
                                        -ExecutionPolicy Bypass, then activate.

  ModuleNotFoundError                   The environment is not active. Run
  (tensorflow/streamlit)                .\.venv310\Scripts\Activate.ps1 and
                                        re-install (step 4).

  Model fails to load                   Use Python 3.10 + TensorFlow 2.10
                                        (.venv310), not Python 3.12.

  FileNotFoundError for                 Run the app from the project root
  app/img/mdc.png                       (step 5).

  Port 8501 is already in use           Use another port:
                                        python -m streamlit run app\app.py
                                        --server.port 8502

  Predictions are all ~20% (random)     The model file is a placeholder.
                                        Download the real one (see above).


CREDITS
-------

  - Original project by Fernanda Rodriguez (mafda) - final project of the
    Complex Data Mining course (MDC013) at Unicamp.
  - Original repository: https://github.com/mafda/knee_OA_dl_app
  - Dataset: https://www.kaggle.com/datasets/shashwatwork/knee-osteoarthritis-dataset-with-severity
  - Techniques: Xception, ResNet50, Inception-ResNet-v2, Grad-CAM.
