# Dataset collection

For MacBook, use `collect_dataset_mac.py` (or add `--manual`). See
[MACBOOK_SETUP.md](MACBOOK_SETUP.md) for setup, camera permissions and copying files.

From this directory, with the virtual environment activated:

```powershell
python collect_dataset.py
```

This starts guided collection with automatic face detection. For unlimited manual
capture with a centered square, use:

```powershell
python collect_dataset_manual.py
```

Both launch the same native interface. Select a subject, session, capture plan and
crop type. A webcam is required; select its index and press Apply if camera 0 fails.
The window displays the actual resolution, which may differ from the request.

| Control | Action |
| --- | --- |
| C | Save one photo; in guided mode it counts toward the selected pose |
| Space | Save up to five photos; guided bursts stop at the pose's target |
| E | Save one extra photo, including after guided completion |
| B | Save five extra photos |
| A | Toggle automatic guided collection with a three-second pose countdown |
| J / K | Select previous / next guided pose |
| S / W | Next / previous session; works for Unknown classmates too |
| N | Create a new Unknown classmate folder after the highest existing number |
| U | Undo the last capture action in the current subject and session |
| Escape | Stop capture; keep successful photos so the pose can be resumed |
| 1–5 | Switch subject |
| Q | Quit |

All controls are also available as buttons or selectors. Keyboard shortcuts are
inactive while a dropdown or camera number input has focus; click the preview to
use them again.

The centered crop does not detect faces. Position one person's face inside its
green square before saving. Detected-face capture requires exactly one current
detection; move bystanders out of frame. Mirroring is applied to the preview and
saved images. Saved crops are always 250 × 250 pixels to match the existing dataset.

## Saved files and resuming

- Guided images: `<script directory>/<person>/s1` through `s6`.
- Unknown: `<script directory>/Unknown/classmate_01`, `classmate_02`, etc.
- Free manual and extra images: `extras/` inside the chosen session folder.
- `_collector_progress.json` records the guided pose for each saved image.
  Keep it with the session so partial bursts and selected poses resume correctly.
- Existing images stay in place. Old sessions have no pose metadata, so their
  starting pose counts are estimated from sequential filenames, five per pose.
  The interface warns when this assumption is used. Inspect old sessions if they
  were interrupted or poses were skipped. No historical pose can be inferred
  reliably from the picture's filename alone.
- Undo applies only to captures made during the current run, in the current folder.
- Interrupted bursts keep successful saves; there is no automatic deletion.
- Extras do not advance guided progress. Include or exclude the `extras/`
  directories deliberately when creating training/validation/test splits. For
  Unknown, split by classmate identity to keep the same person's images out of
  multiple splits.

## Preview quality

Controls use native Windows text and are separate from the video. The preview
fits inside its panel without enlarging low-resolution webcam pixels. The camera
requests 1280 × 720 MJPEG at 30 FPS; the driver chooses supported settings.
Try 1920 × 1080 with Apply if supported. Low light, high camera gain, missed focus
or dirty lenses can still cause grain or blur. Use soft front lighting in ordinary
sessions, keep the lens clean, and avoid digital zoom. Dim/backlit sessions are
deliberately harder conditions. The collector does not smooth saved images or
invent detail to conceal sensor noise.

Dependencies are `opencv-python` and NumPy, plus Tkinter supplied by the standard
Windows Python installer. OpenCV 4 is required for Haar face detection:

```powershell
python -m pip install "opencv-python<5"
```

Camera-free regression checks:

```powershell
python -m unittest test_collector -v
```
