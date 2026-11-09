# Reproduction notes

Every attempt is kept here, including the failed ones. Full outputs are in `logs/`.

## Attempt 1: install the authors' requirements
- **Command:** `pip install -r requirements.txt` (pins `tensorflow-gpu==1.13.0`, `keras==2.2.4`, `sklearn`)
- **Outcome:** `<paste the error, e.g. "No matching distribution found for tensorflow-gpu==1.13.0">`
- **Reason:** TensorFlow 1.13 was released for Python ≤ 3.7. Colab runs Python `<3.x>`.
- **Log:** `logs/attempt1_pip_install.txt`

## Attempt 2: run the authors' notebook with Colab's TensorFlow 2
- **Command:** run `Code/main.ipynb` cells 1–2 as the README says
- **Outcome:** `<paste, e.g. AttributeError: module 'tensorflow' has no attribute 'ConfigProto'; ModuleNotFoundError: No module named 'keras.utils.np_utils'>`
- **Reason:** the code uses TensorFlow 1 / standalone Keras 2.2 APIs.
- **Log:** `logs/attempt2_original_imports.txt`

## Attempt 3: port to TensorFlow 2 (works)
- **Changes** (diff in `logs/attempt3_tf2_port.diff`):
  - `keras.utils.np_utils.to_categorical` → `tensorflow.keras.utils.to_categorical`
  - `keras.layers / models / optimizers` → `tensorflow.keras.*`
  - `SGD(lr=lr)` → `SGD(learning_rate=lr)`
  - `model.predict(...)` → `model.predict(..., verbose=0)`
  - TF1 session cell removed
- **Outcome:** `<runs; first round accuracy ...>`

## Other issues found
- `main.ipynb` sets `CUDA_VISIBLE_DEVICES="2"`. Colab has only GPU 0, so this line would silently force CPU. I removed it.
- `Code/README.md` lists `utils.py`, but that file is not in the repository. Nothing imports it, so it has no effect.
- No random seed is set anywhere, so the client data split differs between runs. I added seeds.
- The README estimates about 24 h for 500 rounds. Measured on Colab T4: `<X>` s per round, so `<T>` rounds were used.
- The authors' saved log has 496 of 500 rounds (a few lines are missing from the notebook output).
- `<anything else: Colab disconnects, re-runs, ...>`

## What was reduced (and why)
- Rounds: 500 → `<T>` (Colab time limit).
- Repetitions: 5 (paper) → 3 seeds.
- All other hyper-parameters are unchanged.
