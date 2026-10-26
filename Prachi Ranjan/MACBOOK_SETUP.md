# Run the collector on a MacBook

Copy these files together to a folder on the Mac:

- `collect_dataset_mac.py`
- `collect_dataset.py` (subject names and guided session settings)
- `collector_gui.py`
- `collector_store.py`
- `requirements-mac.txt`

Copy the person folders and `_collector_progress.json` files too if you want to
continue an existing dataset. Create a fresh Python environment on the Mac;
the Windows `.venv` folder cannot be reused.

Install Python 3 with Tk support. The [python.org macOS installer](https://www.python.org/downloads/macos/)
provides native builds for supported Apple Silicon and Intel Macs. See the
[Python macOS guide](https://docs.python.org/3/using/mac.html) and
[Tcl/Tk information](https://www.python.org/download/mac/tcltk/).

Open Terminal in the folder containing the files and run:

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-mac.txt
python -m tkinter
```

Close the Tk test window, then start the collector:

```sh
python collect_dataset_mac.py
```

For manual capture by default:

```sh
python collect_dataset_mac.py --manual
```

You can switch between guided and manual capture inside either window. Use C for
one image, Space for a burst, E for one extra image, and B for an extra burst.
Select Unknown (5) and press N or click New Unknown session / classmate to create
a new folder. S/W navigate sessions for registered people and Unknown classmates.
Command-Q, Q, and the close button exit cleanly; Escape stops capture. Click the
preview to restore keyboard shortcuts after using a selector. Trackpad scrolling
works over the controls panel. The window adapts to screen height, and the sidebar
scrollbar keeps controls available on smaller MacBook displays.

## Camera

The Mac path uses OpenCV's [AVFoundation backend](https://docs.opencv.org/4.x/d4/d15/group__videoio__flags__base.html),
then automatic backend selection if needed. It lets macOS choose the pixel format
instead of requesting the Windows camera format. The requested starting resolution
is 1280 × 720; the actual resolution is displayed. No artificial video enlargement
or smoothing is applied to saved crops.

Allow camera access when macOS asks. If access was denied, check System Settings
→ Privacy & Security → Camera for the app launching Python (for example Terminal
or your editor), then quit and reopen that app. See
[Apple's camera access instructions](https://support.apple.com/en-gb/guide/mac-help/mchlf6d108da/mac).

If the wrong camera is selected, change the index and click Apply, or start with:

```sh
python collect_dataset_mac.py --camera 1
```

Close other programs using the camera. Changing the camera index can also help
when an external camera or iPhone camera is available. Do not assume every MacBook
supports the same resolution; the camera chooses supported settings.

## Saving and portability

By default images are saved beside the script, with the same registered-person,
Unknown, session, and `extras/` layout as the Windows collector. Saved crops remain
250 × 250 pixels, so the two versions produce compatible datasets.

For a separate location:

```sh
python collect_dataset_mac.py --data-dir ~/Pictures/face_dataset
```

Keep progress files with the images when moving between machines. Run only one
collector against a dataset folder at a time. See `COLLECTOR_README.md` for pose
tracking, old dataset assumptions, undo, and extra-image behavior.

Mac-specific camera selection and scrolling are checked with simulated backends
on Windows. A real MacBook camera and macOS permission prompt still require a
hardware check on the Mac.
