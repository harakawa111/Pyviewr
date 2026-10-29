# Pyviewr

Lightweight desktop viewer for Basler cameras. Preview, still capture, video recording, and dual-camera sync via GigE Action Commands — nothing else.

Target camera: [Basler ace 2 a2A2840-67g5mUV](https://docs.baslerweb.com/a2a2840-67g5muv) (Mono / 5GigE). Other pylon GigE cameras may work.

Inspired by [pylon Viewer](https://www.baslerweb.com/en/software/pylon/pylon-viewer/), but intentionally minimal.

## Features

- Live preview (full frame fitted to the window)
- Still capture (PNG)
- Video recording (MP4 / MPEG-4)
- Easy save-folder selection (persisted in `config.json`)
- Dual-camera synchronized still (and aligned video start) via GigE [Action Commands](https://docs.baslerweb.com/action-commands)
- Optional 10-second timer for still and record start (Off by default; Stop is always immediate)

Not included: feature tree, color calibration, sharpness tools, bandwidth manager, vTools / image processing.

## Requirements

1. **Basler pylon Software Suite 7.1+** (Windows / Linux)  
   Download: https://www.baslerweb.com/en/software/pylon/
2. **Python 3.11+**
3. For 5GigE cameras: a suitable NIC and network setup (Jumbo frames recommended)

### Recording PC requirements

For the target 2840×2840 Mono8 camera at the default 67 fps, the raw input is
about 540 MB/s per camera (about 1.08 GB/s for two cameras).

- **Lossless (FFV1, `.mkv`)** — frames are encoded by `ffmpeg` while
  recording, so the disk only sees ~65 MB/s per camera (~1 MB per frame).
  After Stop, only a fast stream copy fixes the timestamps (~1–2 min for a
  3-minute recording). Requires `ffmpeg` on `PATH`; without it, the raw
  spool path below is used.
- **MP4 (lossy)** — raw frames are spooled to disk and the MP4 is created
  after recording stops, so the disk must sustain the full raw rate.

Verified: two cameras at ~66.7 fps for 3 minutes, no dropped frames, on an
i9-14900K with 32 GB RAM, saving Lossless to a SATA SSD.

- Lossless: about 9 GB per camera for every 3 minutes; a separate data
  SSD (not the OS disk) is recommended
- MP4: NVMe SSD with sustained sequential writes of at least 700 MB/s for
  one camera or 1.3 GB/s for two cameras, and about 65 GB of raw spool
  space per camera for every two minutes
- Many-core CPU (two live FFV1 encoders each use half the cores)
- One 5GigE port per camera with **Jumbo frames (MTU 9000)**; Pyviewr sets
  the camera packet size to match the NIC MTU on connect
- Linux: raise the socket receive buffer, otherwise frames arrive
  incomplete while the CPU is busy encoding:

  ```bash
  sudo sh -c 'printf "net.core.rmem_max=33554432\nnet.core.rmem_default=33554432\n" > /etc/sysctl.d/60-basler-gige.conf'
  sudo sysctl --system
  ```

Actual requirements vary with ROI, pixel format, and frame rate. Confirm
sustained disk throughput before long recordings.

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
   The **67 FPS** button sets 67 fps together with the longest exposure that
   still sustains it; the exposure slider keeps its full range.
3. **Timer** — choose `Timer: Off` (immediate) or `Timer: 10s`. Applies to Still and Record start. Use **Cancel timer** to abort. Stop is always immediate.
4. **Still** — saves `camN_YYYYMMDD_HHMMSS_mmm.png` into the save folder.  
   With two cameras, exposure is triggered by a single GigE Action Command.
5. **Record** / **Stop** — writes `camN_....mp4` (MPEG-4, lossy). Check
   **Lossless** to write `camN_....mkv` (FFV1, bit-exact grayscale — same
   quality as stills). The file frame rate is the rate actually captured, so
   playback runs in real time. With two cameras, the start is aligned with an
   Action Command, then free-run continues. After Stop, a summary line shows
   per camera: fps, frames, `dropped` (camera frames the PC missed), and the
   cause — `incomplete` (network packets lost), `no buffer` (PC too slow to
   take frames), `disk waits` (disk too slow).
6. **Save folder…** — choose where files are written (stored in `config.json`).

Default save directory: `~/Pictures/Pyviewr/`

## Dual-camera notes

- Both cameras should be on the same GigE / 5GigE network segment.
- Sync uses software Action Commands (no trigger cable).
- This is not PTP Scheduled Action Command precision; for tighter sync use PTP + Scheduled Action Commands or hardware trigger outside this app.

## License

MIT
