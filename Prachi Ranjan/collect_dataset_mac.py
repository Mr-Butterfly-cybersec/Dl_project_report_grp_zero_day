"""MacBook entry point: guided, automatic, and unlimited manual face collection.

    python3 collect_dataset_mac.py
    python3 collect_dataset_mac.py --manual
    python3 collect_dataset_mac.py --camera 1 --data-dir ~/Pictures/face_dataset

Keep collect_dataset.py, collector_gui.py, and collector_store.py beside this file.
"""
import argparse
from pathlib import Path
import sys


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description='Face Dataset Studio for MacBook')
    parser.add_argument('--manual', action='store_true',
                        help='Start with unlimited manual capture and a centered square crop.')
    parser.add_argument('--camera', type=int, default=0, metavar='INDEX',
                        help='Camera index, default 0; can also be changed inside the GUI.')
    parser.add_argument('--data-dir', type=Path, default=Path(__file__).resolve().parent,
                        help='Dataset directory, default the directory containing this script.')
    args = parser.parse_args(argv)
    if args.camera < 0:
        parser.error('--camera must be zero or a positive number')
    args.data_dir = args.data_dir.expanduser().resolve()
    return args


def main(argv=None):
    args = parse_args(argv)
    try:
        from collect_dataset import PEOPLE, SESSIONS, UNKNOWN_STEPS, UNKNOWN_ENV
        from collect_dataset import IMAGE_SIZE, SHOTS_PER_STEP, BURST_GAP, AUTO_DELAY
        from collector_gui import run
    except ModuleNotFoundError as exc:
        if exc.name in {'tkinter', '_tkinter'}:
            print('Tkinter is missing. Use a Python installation with Tcl/Tk support, '
                  'such as the macOS installer from https://www.python.org/downloads/macos/.\n'
                  'Check it with: python3 -m tkinter', file=sys.stderr)
        elif exc.name in {'cv2', 'numpy'}:
            print('Install the dependencies first:\n'
                  'python3 -m pip install -r requirements-mac.txt', file=sys.stderr)
        else:
            print(f'Missing companion module: {exc.name}. Copy the collector Python files '
                  'into the same folder.', file=sys.stderr)
        return 1
    if sys.platform != 'darwin':
        print('Running outside macOS; the shared GUI will use this computer\'s camera backend.')
    print(f'Dataset directory: {args.data_dir}')
    run(args.data_dir, PEOPLE, SESSIONS, UNKNOWN_STEPS, UNKNOWN_ENV,
        image_size=IMAGE_SIZE, shots=SHOTS_PER_STEP, gap=BURST_GAP,
        delay=AUTO_DELAY, manual=args.manual, camera_index=args.camera)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
