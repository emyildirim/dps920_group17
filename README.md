# dps920_group17

## Setup

This project requires **Python 3.8**. Newer Python versions (3.10+) are not
compatible with some pinned dependencies (`eventlet`, `tensorflow==2.3.0`) and
will cause the driving simulator connection in `TestSimulation.py` to fail.

1. Create a virtual environment using Python 3.8:
   ```bash
   python3.8 -m venv venv
Activate it:

Windows: venv\Scripts\activate
Mac/Linux: source venv/bin/activate
Install all dependencies from requirements.txt:


pip install -r requirements.txt