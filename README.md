# OpenEyes 👀

A real‑time attention and drowsiness monitoring tool using MediaPipe, OpenCV, and pyttsx3. OpenEyes automatically calibrates baseline metrics and detects:

- **Drowsiness** (continuous eye closure)  
- **Yawning** (mouth aspect ratio spikes)  
- **Head Turn** (sustained yaw angle)  
- **Head Nod** (state‑machine detection of pitch changes)  
- **Blink Rate (BRR)** (blinks per minute)  

It provides a rich, interactive UI with sliders, keyboard controls, real‑time overlays, voice alerts, and CSV logging.

---

## 🚀 Features

- **Auto Calibration**  
  Captures `EAR`, `MAR`, `Yaw`, and `Pitch` baselines & standard deviations.

- **Multi‑Mode Detection**  
  - **Drowsiness**: triggers after N seconds of eye closure.  
  - **Yawning**: based on mouth aspect ratio above threshold.  
  - **Head Turn**: sustained deviation in yaw angle.  
  - **Head Nod**: pitch‑down → pitch‑up → reset state machine.  
  - **Blink Rate**: calculates blinks per minute over a sliding window.

- **Interactive UI**  
  - **Sliders** to tune all thresholds and durations on the fly.  
  - **Keyboard shortcuts** to toggle overlays, re‑calibrate, mute TTS, show help/BRR panel.  
  - **Real‑time overlays**: mesh, status icons, progress bars, numeric panels, FPS.

- **Voice Alerts**  
  - Configurable cooldown and mute toggle.  
  - Alerts for each event: “Drowsiness detected”, “Yawning detected”, etc.

- **CSV Logging**  
  - Logs every frame: timestamp, raw metrics, windowed proportions, BRR, composite score.

---

## 📦 Installation

1. **Clone the repository**  
```bash
git clone https://github.com/yaninsanity/openeyes.git
cd openeyes

python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python3 main.py
```
### 🎮 Interactive Controls

#### Sliders (Trackbars)

| Slider    | Description                                           |
|-----------|-------------------------------------------------------|
| **EARΔ**  | Eye Aspect Ratio baseline offset (lower bound)        |
| **MARΔ**  | Mouth Aspect Ratio baseline offset (upper bound)      |
| **PERC%** | Drowsiness windowed‑proportion threshold (0–1)        |
| **Yawσ\***| Dynamic yaw threshold multiplier                      |
| **Pitσ\***| Dynamic pitch threshold multiplier                    |
| **DrowsyS** | Seconds of continuous eye‑closure → drowsiness     |
| **YawTime** | Seconds of sustained head turn → alert              |
| **NodTime** | Seconds between down/up nod to count as a “nod”     |
| **BRRwin**  | Window length for blink‑rate (BRR) calculation (s)  |

#### Keyboard Shortcuts

| Key          | Action                                |
|--------------|---------------------------------------|
| `c`          | Re‑calibrate baselines                |
| `h`          | Toggle help overlay                   |
| `l`          | Toggle face‑mesh overlay              |
| `f`          | Toggle FPS display                    |
| `p`          | Toggle thresholds panel               |
| `b`          | Toggle BRR (blink rate) panel         |
| `t`          | Toggle mute (disable/enable voice)    |
| `q` / `Esc`  | Quit the application                  |

---

### ⚙️ Command‑Line Arguments

Run with:

```bash
python main.py \
  --calib_delay 3.0 \
  --drowsy_sec 1.5 \
  --perclos_percent 0.4 \
  --log_csv my_log.csv
```

| Argument             | Type   | Default       | Description                                      |
|----------------------|--------|---------------|--------------------------------------------------|
| `--window_size`      | int    | 60            | Number of frames in sliding window              |
| `--calib_frames`     | int    | 120           | Frames to collect during calibration            |
| `--calib_delay`      | float  | 2.0           | Pre-calibration countdown (seconds)             |
| `--drowsy_sec`       | float  | 2.0           | Seconds of continuous eye-closure → drowsiness  |
| `--brr_window`       | float  | 60.0          | Blink-rate window (seconds)                     |
| `--ear_delta`        | float  | 0.15          | EAR threshold offset                            |
| `--mar_delta`        | float  | 0.20          | MAR threshold offset                            |
| `--yaw_delta`        | float  | 15.0          | Static yaw threshold (degrees)                  |
| `--pitch_delta`      | float  | 15.0          | Static pitch threshold (degrees)                |
| `--yaw_time_thresh`  | float  | 1.0           | Seconds of sustained head turn → alert          |
| `--nod_time_thresh`  | float  | 1.0           | Seconds between down/up nod → count as nod      |
| `--yaw_std_factor`   | float  | 2.0           | Dynamic yaw threshold multiplier                |
| `--pitch_std_factor` | float  | 2.0           | Dynamic pitch threshold multiplier              |
| `--perclos_percent`  | float  | 0.30          | Windowed-proportion threshold for drowsiness    |
| `--mar_percent`      | float  | 0.20          | Windowed-proportion threshold for yawning       |
| `--yaw_percent`      | float  | 0.20          | Windowed-proportion threshold for head turn     |
| `--pitch_percent`    | float  | 0.20          | Windowed-proportion threshold for nodding       |
| `--smooth`           | float  | 0.6           | Exponential smoothing factor for composite score |
| `--alert_cooldown`   | float  | 5.0           | Minimum seconds between voice alerts            |
| `--log_csv`          | str    | attention_log.csv | Path to output CSV log                        |


## 📄 CSV Log Format

Each row in the output CSV (default: `attention_log.csv`) has the following columns:

| Column | Type    | Description                                                          |
|:-------|:--------|:---------------------------------------------------------------------|
| `ts`   | `str`   | Timestamp in `HH:MM:SS`                                              |
| `ear`  | `float` | Eye Aspect Ratio measured this frame                                 |
| `mar`  | `float` | Mouth Aspect Ratio measured this frame                               |
| `yaw`  | `float` | Yaw angle (°) measured this frame                                    |
| `pit`  | `float` | Pitch angle (°) measured this frame                                  |
| `pe`   | `float` | Sliding‑window proportion of eye‑closure events (0 – 1)               |
| `pm`   | `float` | Sliding‑window proportion of yawning events (0 – 1)                   |
| `py`   | `float` | Sliding‑window proportion of head‑turn events (0 – 1)                 |
| `pp`   | `float` | Sliding‑window proportion of nod events (0 – 1)                       |
| `brr`  | `float` | Blink Rate in beats per minute (BPM) over the configured window       |
| `score`| `float` | Composite attention score (0 – 1) combining all four flags            |

## 🚧 Contributing
We welcome improvements! Here are some ideas:

- New Detection Modes
Add blink fatigue, gaze tracking, emotion recognition, etc.
- UI Enhancements
Replace OpenCV windows with a modern GUI (Qt, DearPyGUI, web dashboard).

- Data Analysis
Provide Jupyter notebooks or dashboards to visualize logged CSV data.

- Performance
GPU acceleration, quantized or native builds, reduce CPU usage.

Please open an Issue or submit a Pull Request. 
📝 License
This project is licensed under the MIT License. See LICENSE for details.