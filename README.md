# swallowsafe-event-classifier

## Overview
SwallowSafe classifies throat activity (talking, drinking/swallowing, coughing, laughing and rest) from three sensors and a microphone sampled at 50 Hz. There are two approaches: a Random Forest on 5-second windows, and a causal 1D CNN that labels every 20 ms sample and can run in real time. Associated with the Figueroa Lab.

## Methods
- `swallowsafe_classification.py` (Random Forest)
  - Preprocessing: demeaning, 0.5–20 Hz Butterworth bandpass on the piezos, and per-block normalization.
  - Per-channel features: mean, std, max, min, RMS, peak-to-peak, FFT energy, dominant frequency, and short-burst RMS max/ratio/std.
  - Events are aligned to the recording protocol (talk / water / cough, each followed by a break).
  - Leave-one-block-out cross-validation. The final model has 300 trees with balanced class weights.
- `swallowsafe_1d_cnn.py` (1D CNN)
  - Input: 2 s of history (100 samples).
  - Causal 2nd-order 0.5–20 Hz bandpass (`sosfilt` with carried state), with normalization statistics taken from training blocks only.
  - Mic handling: a 5th input marks mic dropouts (raw value < 9 or > 100, held for 10 samples).
  - Model: dilated residual Conv1d blocks (width 64, dilations 1/2/4/8) with mean, max and recent-sample pooling.
  - Training: AdamW with a OneCycle schedule, 30 epochs, class-weighted cross-entropy with label smoothing 0.05. Augmentation: random gain (0.8–1.2), Gaussian noise and random mic dropout (p = 0.15).
  - Evaluation: 5-fold blocked cross-validation with whole blocks held out. Frame-level and event-level metrics (events shorter than 0.3 s are ignored; an event counts as detected at ≥30% overlap).
  - Includes a streaming classifier and a check that streaming predictions match offline predictions.
- Caveats stated in the code:
  - Break makes up about 71% of samples.
  - Cough appears only in blocks 1–14 and 29, and Laugh only in blocks 15–28 and 30. The model may partly learn the session rather than the class, so Cough/Laugh scores are likely optimistic.

Results: The code prints accuracy and macro-F1 at run time but stores no results.

## Tools
Python, PyTorch, SciPy, NumPy, pandas, scikit-learn, Matplotlib, seaborn, gspread (Google Sheets), Google Colab.

## Status
In progress.

**Data:** not included. Recordings come from one participant, a member of the project team. **Some event labels were produced with AI assistance and are being edited currently.** 

**Paths:** Google Drive paths were replaced with placeholders under `/path/to/data/SwallowSafe/`.

**Notes:** The `.py` files are exports of Google Colab notebooks. Lines starting with `!` (and the bare `pip install` line, where present) are notebook shell commands, and `drive.mount(...)` calls are Colab-only. Run the files cell by cell in a notebook, not as plain scripts. `swallowsafe_1d_cnn.py` calls `load_blocks_from_drive_sheets_by_name`, which was defined in a notebook cell not included in this export.
