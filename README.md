rPPG Real-Time Heart Rate Monitor
This project implements a remote Photoplethysmography (rPPG) system that estimates Heart Rate (BPM) and Blood Volume Pulse (BVP) from a live camera feed. It uses a PhysNet 3D-CNN architecture and features a clinical-grade visualization dashboard.

for more information about rPPG, please check out this repository:
github.com/nasir6/rPPG

Create and Activate Virtual Environment
It is recommended to use a virtual environment to keep your system dependencies clean.

for Linux
``
python3 -m venv rppg_env
source rppg_env/bin/activate
``
for Windows
``
python -m venv rppg_env
rppg_env\Scripts\activate
``

Install Necessary Libraries
Install the core dependencies for computer vision, deep learning (PyTorch), and signal processing.

``
pip install torch torchvision numpy opencv-python matplotlib scipy pillow
``

Run the System
Start the application by pointing to your camera source.

``
python3 run.py --source 0 --frame-rate 30
``
