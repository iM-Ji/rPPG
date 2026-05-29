---

# rPPG: Real-Time Heart Rate Monitor

This project implements a **remote Photoplethysmography (rPPG)** system that estimates Heart Rate (BPM) and Blood Volume Pulse (BVP) from a live camera feed. It utilizes a **PhysNet 3D-CNN architecture** and features a clinical-grade visualization dashboard.

> [!TIP]
> For more in-depth information about the underlying rPPG research and models, please visit the original repository:
> 🔗 **[github.com/nasir6/rPPG](https://github.com/nasir6/rPPG)**

---

## 🛠 1. Create and Activate Virtual Environment

It is highly recommended to use a virtual environment to ensure system dependencies remain clean and isolated.

### **For Linux (Fedora/Ubuntu)**

```bash
python3 -m venv rppg_env
source rppg_env/bin/activate

```

### **For Windows**

```bash
python -m venv rppg_env
rppg_env\Scripts\activate

```

---

## 📦 2. Install Necessary Libraries

Install the core dependencies for computer vision, deep learning (PyTorch), and advanced signal processing.

```bash
pip install torch torchvision numpy opencv-python matplotlib scipy pillow

```

---

## 🚀 3. Run the System

Start the application by targeting your local camera source. Ensure you are in a well-lit environment for the best detection results.

```bash
python3 run.py --source 0 --frame-rate 30

```

