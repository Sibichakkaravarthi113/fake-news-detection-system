# FakeNewsDetect - Backend + Desktop Demo

## Prereqs
- Python 3.9+ (recommended)
- If using GPU: install appropriate NVIDIA driver + CUDA or use conda package for pytorch-cuda.
- Recommended: create and activate a conda env.

## Install
pip install -r requirements.txt
(For torch, follow official install instructions: https://pytorch.org)

## Ensure models exist
Place these files/folder inside `models/`:
- models/tfidf.joblib
- models/LogisticRegression.joblib
- models/NaiveBayes.joblib
- models/SVM.joblib
- models/xlm_model/  (folder saved by transformers Trainer)

If you don't have these, run training scripts in `src/` to produce them.

## Run backend
uvicorn src.api:app --reload

Open: http://127.0.0.1:8000/docs for Swagger UI.

## Run GUI
python frontend/app_gui.py

Enter news text, click "Check News", view results.

## Notes
- Wikipedia results depend on internet connectivity.
- For production, replace Wikipedia heuristics with a robust fact-check API and entity matching.
