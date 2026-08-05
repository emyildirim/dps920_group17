# Evaluation and Simulator Testing Report

## Scope

This report evaluates the trained CNN in `model.h5`. The model receives a centre-camera image and predicts one continuous steering angle. This is a regression task.

## Offline evaluation method

The evaluation script recreates the notebook's data preparation:

1. Load centre-camera paths and steering values.
2. Randomly remove 80% of rows whose steering value is exactly `0.0`.
3. Recreate the 80/20 split with random seed 42.
4. Use the validation rows only. Do not augment these images.
5. Apply the training crop, RGB-to-YUV conversion, `200 x 66` resize, normalization, and Gaussian blur.
6. Compare model predictions with the recorded steering values.

This recreated validation split is useful, but it is not a fully independent test set. Adjacent simulator frames are similar. A complete recording session that was not used during training would be a stronger final test.

## Metrics

The generated results are in `results/metrics.txt` and `results/metrics.json`.

The zero-steering baseline always predicts `0.0`. The CNN must improve on this baseline to show that it learned more than the dataset's common straight-driving value.

The evaluation used 1,609 validation images. All image paths resolved correctly.

| Metric | CNN | Zero-steering baseline |
| --- | ---: | ---: |
| MAE | 0.047490 | 0.084561 |
| MSE | 0.005474 | 0.015295 |
| RMSE | 0.073987 | 0.123674 |
| R-squared | 0.641732 | -0.001046 |

The CNN reduces RMSE by **40.18%** compared with the zero-steering baseline. This result shows that the model learned useful steering information instead of only predicting a value near zero.

The evaluation also reports separate errors for:

- Near-straight driving: `|angle| < 0.05`
- Small turns: `0.05 <= |angle| < 0.20`
- Medium turns: `0.20 <= |angle| < 0.40`
- Sharp turns: `|angle| >= 0.40`

| Steering range | Samples | MAE | RMSE |
| --- | ---: | ---: | ---: |
| Near straight | 708 | 0.030435 | 0.047834 |
| Small turn | 751 | 0.045258 | 0.057445 |
| Medium turn | 127 | 0.109254 | 0.131575 |
| Sharp turn | 23 | 0.304315 | 0.330413 |

The error increases with turn strength. Sharp turns are the main model weakness. Only 23 sharp-turn images are in the validation split, so the simulator test must confirm whether the car turns early and strongly enough.

## Generated figures

- `results/actual_vs_predicted.png`
- `results/error_histogram.png`
- `results/steering_sequence.png`
- `results/error_by_turn_range.png`
- `results/worst_predictions.png`

## Simulator test procedure

1. Start `TestSimulation.py` with the trained model.
2. Open the Udacity simulator.
3. Use the same track used to collect the data.
4. Select **Autonomous Mode**.
5. Do not give manual steering input.
6. Record the simulator and terminal during at least one complete lap.

### Acceptance criteria

- The car completes one full lap without manual steering.
- The car stays on the road and avoids barriers.
- It handles straight sections and left and right turns.
- It does not show severe left-right oscillation.
- Its speed stays controlled.

## Final simulator result

**Status: failed - the current model does not complete a full lap.**

The test used the official Udacity Term 1 Linux simulator, Track 1, Autonomous Mode, target speed 6 MPH, and minimum speed 4 MPH. The inference server connected successfully and continuously returned steering and throttle commands. The car stayed on the road during the first sections, drifted toward the road edge near a major curve, crossed the curb, and stopped. It did not complete one lap.

Lower speed and small steering calibration changes did not produce a full lap. Recovery-aware synthetic retraining was also tested, but it did not produce a reliable lap and was discarded. A successful screen recording cannot be claimed yet.

### Required next action

1. Record three to five slow, clean centre-camera passes through the failure curve.
2. Record short recovery examples that start near each road edge and steer smoothly back to the centre.
3. Remove or correct the notebook's panning and rotation augmentations because they change image geometry without changing the steering label.
4. Retrain, then repeat the full-lap test at low speed before testing at a higher speed.

## Main risks and limits

- A random frame split can place very similar adjacent frames in training and validation.
- The dataset has many straight-driving frames. Removing excess exact-zero rows reduces this bias but does not make the distribution uniform.
- Horizontal panning and rotation in the notebook do not adjust the steering label. This can add label noise.
- Simulator inference must use RGB input. OpenCV training images were converted from BGR to RGB before preprocessing.
