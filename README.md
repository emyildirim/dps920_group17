# DPS920 Group 17 — Self-Driving Car Using a CNN

This project trains a convolutional neural network to predict a continuous steering angle from the Udacity self-driving car simulator's center-camera image. The trained model is then evaluated on labelled images and connected to the simulator for autonomous driving.

## Team responsibilities

| Member | Main contribution |
|---|---|
| `emyildirim` | Dataset preparation, balancing, augmentation, and batching |
| `AbdulmuhsinB` | Image preprocessing, CNN training, and saved model |
| Ahmed Shaikh (`Winter`) | Offline evaluation, simulator inference/testing, results, and testing documentation |

## Repository files

- `data_preprocessing.ipynb` — data inspection, balancing, augmentation, preprocessing, batching, CNN construction, and training.
- `fix_paths.py` — changes machine-specific image paths in the Udacity CSV to relative `IMG/` paths.
- `model.h5` — final trained Keras model.
- `preprocessing.py` — shared training-compatible image preprocessing used during evaluation and live inference.
- `evaluation.py` — calculates metrics and produces evaluation plots.
- `TestSimulation.py` — connects the model to the simulator in Autonomous Mode.
- `requirements-testing.txt` — additional evaluation and simulator dependencies.
- `results/` — generated metrics, predictions, and figures.

## Dataset

The recorded dataset is too large for GitHub and is stored on Google Drive:

https://drive.google.com/file/d/18eGWZ25Gu00CrdBEzYPy1bDvhWF2gn0L/view?usp=share_link

After downloading and extracting it, the dataset folder should contain:

```text
dataset/
├── driving_log.csv
└── IMG/
    ├── center_....jpg
    └── ...
```

## Approach

1. Record simulator camera images and steering values while manually driving in both directions.
2. Use the center camera and continuous `Steering` value.
3. Plot the steering-angle histogram and randomly reduce the excess exact-zero samples.
4. Split training and validation data before augmentation.
5. Randomly apply flipping, brightness, zoom, panning, and rotation only to training data.
6. Crop rows `60:135`, convert RGB to YUV, resize to `200 × 66`, normalize, and apply Gaussian blur.
7. Train the Nvidia-style CNN and save it as `model.h5`.
8. Evaluate predictions on labelled images.
9. Run the model in the simulator and record a successful autonomous lap.

## Environment setup

Use the same virtual environment in which `model.h5` was trained whenever possible. This avoids TensorFlow/HDF5 compatibility problems.

### Windows

Prefer Python 3.10 or 3.11 (TensorFlow support is more reliable than on 3.13):

```powershell
python3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements-testing.txt
```

If the model was trained in a different existing course environment, activate that environment and install only the missing simulator packages.

### Download the dataset

```powershell
.\.venv\Scripts\Activate.ps1
python .\scripts\download_dataset.py
```

Or place an extracted folder at `dataset/driving_log.csv` + `dataset/IMG/`.

## Offline evaluation

For a trustworthy final evaluation, record or reserve a complete driving session that was not used for training. Using the training dataset is acceptable as an integration check but produces optimistic metrics.

Quick run (after dataset download):

```powershell
.\scripts\run_evaluation.ps1
```

Or manually:

```powershell
python evaluation.py `
  --model model.h5 `
  --csv ".\dataset\driving_log.csv" `
  --image-root ".\dataset" `
  --output-dir results
```

A quick trial on a smaller reproducible subset:

```powershell
python evaluation.py --model model.h5 --csv "C:\path\to\dataset\driving_log.csv" --image-root "C:\path\to\dataset" --max-samples 500
```

The script creates:

- `results/metrics.json`
- `results/predictions.csv`
- `results/actual_vs_predicted.png`
- `results/error_histogram.png`
- `results/steering_sequence.png`
- `results/error_by_turn_range.png`
- `results/worst_predictions.png`

The report includes MAE, MSE, RMSE, R², tolerance accuracy, an always-zero baseline, and separate errors for straight, small, medium, and sharp turns.

## Simulator testing

1. Activate the same environment used for training.
2. Start the inference server:

```powershell
.\scripts\run_simulator.ps1 9 6
```

Or:

```powershell
python TestSimulation.py --model model.h5 --target-speed 9 --minimum-speed 6
```

3. Launch the Udacity simulator with the same resolution and graphics quality used during data collection.
4. Select **Autonomous Mode**.
5. Observe the terminal and vehicle behaviour.
6. Record at least one complete autonomous lap with no manual steering corrections.

Begin with a low speed. Increase it only after the car stays stable:

```powershell
python TestSimulation.py --model model.h5 --target-speed 9 --minimum-speed 6
```

Useful tuning options:

- Reduce `--target-speed` if the car leaves the road on curves.
- Increase `--steering-slowdown` if it enters curves too quickly.
- Lower `--smoothing` if steering oscillates.
- Raise `--smoothing` toward `1.0` if it turns too slowly because of excessive smoothing.

## Evaluation criteria

The final simulator test should document whether the vehicle:

- Completes a full lap without manual correction.
- Remains inside the road boundaries.
- Handles both left and right turns.
- Avoids excessive left-right oscillation.
- Maintains a controlled speed.

## Major challenges and solutions

### Excess straight-driving data

The raw histogram contained a large concentration near steering `0.0`, which could bias the model toward driving straight. Excess exact-zero examples were randomly removed while preserving curved-road examples.

### Machine-specific image paths

The simulator CSV stored absolute paths from the collection computer. `fix_paths.py` converts them to portable relative `IMG/<filename>` paths, while the evaluation script also provides path fallbacks.

### Training/inference preprocessing mismatch

A model can perform poorly if live images are cropped, colour-converted, resized, or normalized differently from training. Both `evaluation.py` and `TestSimulation.py` import the same `preprocessing.py` implementation.

### Simulator steering jitter

Live predictions can fluctuate between consecutive frames. The simulator script includes configurable exponential steering smoothing and reduces target speed as turns become sharper.

## Final results

Do not copy training loss into this section as the final test result. After running `evaluation.py`, summarize the values from `results/metrics.json` and compare them with the always-zero baseline.

- Evaluation dataset/session: **To be filled after execution**
- Number of evaluated images: **To be filled after execution**
- Model MAE: **To be filled after execution**
- Model RMSE: **To be filled after execution**
- Always-zero baseline MAE: **To be filled after execution**
- Full autonomous lap completed: **Yes / No**
- Screen-recording link: **To be added**
