# Pyviewr

Lightweight desktop viewer for Basler cameras. Preview, still capture, video recording, and dual-camera sync via GigE Action Commands — nothing else.

Target camera: [Basler ace 2 a2A2840-67g5mUV](https://docs.baslerweb.com/a2a2840-67g5muv) (Mono / 5GigE). Other pylon GigE cameras may work.

Inspired by [pylon Viewer](https://www.baslerweb.com/en/software/pylon/pylon-viewer/), but intentionally minimal.

## Features

- Live preview (full frame fitted to the window)
- Still capture (PNG)
- Video recording (AVI / MJPG)
- Easy save-folder selection (persisted in `config.json`)
- Dual-camera synchronized still (and aligned video start) via GigE [Action Commands](https://docs.baslerweb.com/action-commands)

Not included: feature tree, color calibration, sharpness tools, bandwidth manager, vTools / image processing.

## Requirements

1. **Basler pylon Software Suite 7.1+** (Windows / Linux)  
   Download: https://www.baslerweb.com/en/software/pylon/
2. **Python 3.11+**
3. For 5GigE cameras: a suitable NIC and network setup (Jumbo frames recommended)

## Install

```bash
# create and activate a venv (recommended)
python -m venv .venv

# Windows
.venv\Scripts\activate

# Linux / macOS
# source .venv/bin/activate

pip install -r requirements.txt
```

## Run

```bash
python -m pyviewr
```

## Usage

1. Click **Refresh** to list cameras.
2. Click **Connect** — opens the first one or two cameras and starts preview.
3. **Still** — saves `camN_YYYYMMDD_HHMMSS_mmm.png` into the save folder.  
   With two cameras, exposure is triggered by a single GigE Action Command.
4. **Record** / **Stop** — writes `camN_....avi` (MJPG). With two cameras, the start is aligned with an Action Command, then free-run continues.
5. **Save folder…** — choose where files are written (stored in `config.json`).

Default save directory: `~/Pictures/Pyviewr/`

## Dual-camera notes

- Both cameras should be on the same GigE / 5GigE network segment.
- Sync uses software Action Commands (no trigger cable).
- This is not PTP Scheduled Action Command precision; for tighter sync use PTP + Scheduled Action Commands or hardware trigger outside this app.

## License

MIT
