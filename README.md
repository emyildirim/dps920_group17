# dps920_group17

Computer Vision final project — CNN steering model for the Udacity self-driving car simulator.

Pinned deps (`eventlet`, `tensorflow==2.3.0`, `python-socketio`) are from ~2021. Use **Windows + Python 3.8**. Newer Python will break the sim connection.

## Requirements

- Python 3.8
- [Udacity Self-Driving Car Simulator](https://github.com/udacity/self-driving-car-sim) (Windows build) — for testing only

## Dataset

Too big for GitHub. Download:

https://drive.google.com/file/d/18eGWZ25Gu00CrdBEzYPy1bDvhWF2gn0L/view?usp=share_link

Extract so you have `dataset/driving_log.csv` and `dataset/IMG/`.

## Setup

```powershell
py -0
py -3.8 -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

If 3.8 isn’t listed, install it from [python.org](https://www.python.org/downloads/release/python-3810/) first.

Use only `requirements.txt`. Don’t mix in other package lists — wrong versions break `TestSimulation.py`.

## Training

Activate the venv, open `data_preprocessing.ipynb`, point it at the dataset, run the cells. Saves `model.h5`.

## Testing

1. `model.h5` in the project root
2. Venv on:

```powershell
python TestSimulation.py
```

You should see something like:

```text
Setting Up ...
* Running on http://127.0.0.1:4567
```

3. Open the simulator → **Autonomous Mode** (connects to `localhost:4567` on its own)
4. Terminal prints `Connected`, then `throttle, steering, speed` — car drives from the model

## Notes

The autonomous run isn’t perfect. The dataset is still skewed toward going straight, so the car often keeps going straight into a steep curve when it should turn. We think trimming/balancing the data more would make it sharper and more accurate.



# An issue and resolution:
So one issue we had was figuring out how to actually implement the model based off the diagram given in Figure 7 of the Final_Project.pdf. To further our understanding we found the paper that we believe the Figure 7 stemmed from: End to End Learning for Self-Driving Cars (https://arxiv.org/pdf/1604.07316) 

By reading this paper I was able to figure out that the first 3 convultional layers would utilise 2x2 stride with a 5x5 kernel. and the last 3 layers would use a 3x3 kernel. And the paper helped me make sense of the other aspects of the diagram. In hindsight I can now see how to read the diagram properly. 

### AI citation
AI tools were used to help with documentation and with some valuation/testing scripts.  Model training and core project code were done by the team.
