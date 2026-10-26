"""Native controls and an unscaled camera preview for both capture entry points."""
import os
import sys
import time
import tkinter as tk
from tkinter import ttk, messagebox, font as tkfont

import cv2

from collector_store import CaptureStore, classmate_indices


def open_camera(index, width=1280, height=720):
    mac = sys.platform == 'darwin'
    if mac:
        backends = (cv2.CAP_AVFOUNDATION, cv2.CAP_ANY)
    elif os.name == 'nt':
        backends = (cv2.CAP_DSHOW, cv2.CAP_ANY)
    else:
        backends = (cv2.CAP_ANY,)
    for backend in backends:
        cap = cv2.VideoCapture(index, backend)
        try:
            if cap.isOpened():
                # Let AVFoundation negotiate its native pixel format on Mac.
                if not mac:
                    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
                cap.set(cv2.CAP_PROP_FPS, 30)
                if not mac:
                    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                # Allow camera startup/permission handling before giving up.
                for _ in range(10 if mac else 1):
                    ok, frame = cap.read()
                    if ok and frame is not None:
                        return cap, frame
                    if mac:
                        time.sleep(0.05)
        except cv2.error:
            pass
        cap.release()
    advice = (' Check System Settings > Privacy & Security > Camera for the app launching Python, '
              'then quit and reopen that app.' if mac else '')
    raise RuntimeError(f'Cannot read camera {index}. Close other camera apps or try another index.' + advice)


def scroll_units(delta, mac=False):
    """Aqua trackpads send small deltas; Windows wheels normally send multiples of 120."""
    if not delta:
        return 0
    divisor = 1 if mac else 120
    amount = max(1, min(10, round(abs(delta) / divisor)))
    return -amount if delta > 0 else amount


def load_detectors():
    if not hasattr(cv2, 'CascadeClassifier') or not hasattr(cv2, 'data'):
        raise RuntimeError('Face detection requires OpenCV 4. Install: python -m pip install "opencv-python<5"')
    root = cv2.data.haarcascades
    frontal = cv2.CascadeClassifier(root + 'haarcascade_frontalface_default.xml')
    profile = cv2.CascadeClassifier(root + 'haarcascade_profileface.xml')
    if frontal.empty():
        raise RuntimeError('Cannot load the frontal face detector. Reinstall opencv-python.')
    return frontal, None if profile.empty() else profile


def detect_faces(frame, frontal, profile):
    h, w = frame.shape[:2]
    scale = min(1.0, 640 / w)
    gray = cv2.cvtColor(cv2.resize(frame, (round(w * scale), round(h * scale))), cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    boxes = list(frontal.detectMultiScale(gray, 1.1, 5, minSize=(30, 30)))
    if not boxes and profile is not None:
        boxes = list(profile.detectMultiScale(gray, 1.1, 4, minSize=(30, 30)))
        if not boxes:
            boxes = [(gray.shape[1] - x - fw, y, fw, fh)
                     for x, y, fw, fh in profile.detectMultiScale(cv2.flip(gray, 1), 1.1, 4, minSize=(30, 30))]
    return [tuple(round(v / scale) for v in box) for box in boxes]


def capture_box(frame, faces, centered=False):
    h, w = frame.shape[:2]
    if centered:
        size = min(300, h, w)
        return ((w - size) // 2, (h - size) // 2, size, size)
    # Refuse ambiguous frames instead of silently saving a bystander's face.
    if len(faces) != 1:
        return None
    x, y, fw, fh = faces[0]
    size = min(round(max(fw, fh) * 1.4), h, w)
    x = max(0, min(w - size, x + fw // 2 - size // 2))
    y = max(0, min(h - size, y + fh // 2 - size // 2))
    return x, y, size, size


def crop_image(frame, box, image_size):
    x, y, w, h = box
    crop = frame[y:y + h, x:x + w]
    if crop.size == 0:
        raise ValueError('Empty capture region.')
    method = cv2.INTER_AREA if w >= image_size[0] else cv2.INTER_CUBIC
    return cv2.resize(crop, image_size, interpolation=method)


class CollectorApp:
    def __init__(self, root, base, people, sessions, unknown_steps, unknown_env,
                 image_size, shots, gap, delay, manual, camera_index=0):
        self.root, self.base = root, base
        self.people, self.sessions = people, sessions
        self.unknown_steps, self.unknown_env = unknown_steps, unknown_env
        self.image_size, self.shots, self.gap, self.delay = image_size, shots, gap, delay
        self.cap = None
        self.detectors = None
        self.frame = self.box = None
        self.photo = None
        self.closed = False
        self.failures = 0
        self.stores = {}
        self.store = None
        self.person = '1'
        self.session = 0
        self.classmate = max(classmate_indices(base / people['5']), default=1)
        self.pose = 0
        self.remaining = 0
        self.burst_step = None
        self.next_shot = 0.0
        self.last_save = -float('inf')
        self.last_ok = time.monotonic()
        self.ready_at = time.monotonic()
        self.auto = tk.BooleanVar(value=False)
        self.plan = tk.StringVar(value='Free manual' if manual else 'Guided poses')
        self.crop_mode = tk.StringVar(value='Centered square' if manual else 'Detected face')
        self.member = tk.StringVar(value=f'1: {people["1"]}')
        self.pose_var = tk.StringVar()
        self.status = tk.StringVar(value='Opening camera...')
        self.camera_info = tk.StringVar()
        self.progress = tk.StringVar()
        self.heading = tk.StringVar()
        self.environment = tk.StringVar()
        self.path_info = tk.StringVar()
        self.face_info = tk.StringVar()
        self.camera_index = tk.StringVar(value=str(camera_index))
        self.resolution = tk.StringVar(value='1280 x 720')
        self._build()
        self.select_context()
        self.reconnect()
        root.protocol('WM_DELETE_WINDOW', self.close)
        root.bind('<KeyPress>', self.on_key)
        self.video.bind('<Button-1>', lambda e: self.video.focus_set())
        if sys.platform == 'darwin':
            root.bind('<Command-q>', lambda e: self.close())
            root.createcommand('tk::mac::Quit', self.close)
        root.after(30, self.tick)

    def _build(self):
        root = self.root
        mac = sys.platform == 'darwin'
        root.title('Face Dataset Studio - Mac' if mac else 'Face Dataset Studio')
        width = min(1200, max(800, root.winfo_screenwidth() - 80))
        height = min(820, max(600, root.winfo_screenheight() - 100))
        root.geometry(f'{width}x{height}')
        root.minsize(min(1000, width), min(700, height))
        style = ttk.Style(root)
        if mac and 'aqua' in style.theme_names():
            style.theme_use('aqua')
        elif not mac and 'vista' in style.theme_names():
            style.theme_use('vista')
        self.ui_font = tkfont.nametofont('TkDefaultFont').copy()
        self.title_font = self.ui_font.copy()
        self.title_font.configure(size=18 if mac else 16, weight='bold')
        self.status_font = self.ui_font.copy()
        self.status_font.configure(weight='bold')
        style.configure('TLabel', font=self.ui_font)
        style.configure('TButton', font=self.ui_font, padding=5)
        style.configure('Title.TLabel', font=self.title_font)
        shell = ttk.Frame(root, padding=12)
        shell.pack(fill='both', expand=True)
        side = ttk.Frame(shell, width=390)
        side.pack(side='left', fill='y', padx=(0, 12))
        side.pack_propagate(False)
        canvas = tk.Canvas(side, width=370, highlightthickness=0)
        scroll = ttk.Scrollbar(side, orient='vertical', command=canvas.yview)
        scroll.pack(side='right', fill='y')
        canvas.pack(side='left', fill='both', expand=True)
        canvas.configure(yscrollcommand=scroll.set)
        sidebar = ttk.Frame(canvas)
        item = canvas.create_window((0, 0), window=sidebar, anchor='nw')
        sidebar.bind('<Configure>', lambda e: canvas.configure(scrollregion=canvas.bbox('all')))
        canvas.bind('<Configure>', lambda e: canvas.itemconfigure(item, width=e.width))
        root.bind('<MouseWheel>', lambda e: canvas.yview_scroll(scroll_units(e.delta, mac), 'units')
                  if e.x_root < side.winfo_rootx() + side.winfo_width() else None)
        ttk.Label(sidebar, text='Dataset Studio', style='Title.TLabel').pack(anchor='w')
        ttk.Label(sidebar, text='Subject').pack(anchor='w', pady=(10, 2))
        cb = ttk.Combobox(sidebar, textvariable=self.member, state='readonly',
                          values=[f'{k}: {name}' for k, name in self.people.items()])
        cb.pack(fill='x')
        cb.bind('<<ComboboxSelected>>', self.change_person)
        ttk.Label(sidebar, textvariable=self.heading, wraplength=350).pack(anchor='w', pady=(10, 4))
        row = ttk.Frame(sidebar)
        row.pack(fill='x')
        ttk.Button(row, text='Previous [W]', command=lambda: self.change_session(-1)).pack(side='left')
        ttk.Button(row, text='Next [S]', command=lambda: self.change_session(1)).pack(side='left', padx=4)
        self.new_button = ttk.Button(sidebar, text='New Unknown session / classmate [N]', command=self.new_classmate)
        self.new_button.pack(fill='x', pady=4)
        ttk.Label(sidebar, textvariable=self.environment, wraplength=350).pack(anchor='w', pady=(2, 6))
        row = ttk.Frame(sidebar)
        row.pack(fill='x')
        for title, var, values in (
            ('Capture plan', self.plan, ['Guided poses', 'Free manual']),
            ('Crop', self.crop_mode, ['Detected face', 'Centered square']),
        ):
            col = ttk.Frame(row)
            col.pack(side='left', fill='x', expand=True, padx=(0, 5))
            ttk.Label(col, text=title).pack(anchor='w')
            combo = ttk.Combobox(col, textvariable=var, values=values, state='readonly', width=18)
            combo.pack(fill='x')
            combo.bind('<<ComboboxSelected>>', self.mode_changed)
        ttk.Label(sidebar, text='Guided pose (J / K)').pack(anchor='w', pady=(8, 2))
        self.pose_combo = ttk.Combobox(sidebar, textvariable=self.pose_var, state='readonly')
        self.pose_combo.pack(fill='x')
        self.pose_combo.bind('<<ComboboxSelected>>', self.change_pose)
        ttk.Label(sidebar, textvariable=self.progress, wraplength=350).pack(anchor='w', pady=5)
        self.progress_bar = ttk.Progressbar(sidebar, maximum=1)
        self.progress_bar.pack(fill='x', pady=(0, 6))
        ttk.Button(sidebar, text='Capture one photo [C]', command=lambda: self.start_capture(1)).pack(fill='x', pady=2)
        ttk.Button(sidebar, text='Capture burst [Space]', command=lambda: self.start_capture(self.shots)).pack(fill='x', pady=2)
        ttk.Button(sidebar, text='Extra photo - any time [E]', command=lambda: self.start_capture(1, extra=True)).pack(fill='x', pady=2)
        ttk.Button(sidebar, text='Extra burst [B]', command=lambda: self.start_capture(self.shots, extra=True)).pack(fill='x', pady=2)
        self.auto_button = ttk.Checkbutton(sidebar, text=f'Auto guided: {self.delay:g}s between poses [A]',
                                           variable=self.auto, command=self.toggle_auto)
        self.auto_button.pack(anchor='w', pady=4)
        row = ttk.Frame(sidebar)
        row.pack(fill='x')
        ttk.Button(row, text='Undo last capture [U]', command=self.undo).pack(side='left')
        ttk.Button(row, text='Stop [Esc]', command=self.stop).pack(side='left', padx=4)
        ttk.Separator(sidebar).pack(fill='x', pady=8)
        row = ttk.Frame(sidebar)
        row.pack(fill='x')
        ttk.Label(row, text='Camera').pack(side='left')
        ttk.Spinbox(row, from_=0, to=9, width=3, textvariable=self.camera_index).pack(side='left', padx=4)
        ttk.Combobox(row, textvariable=self.resolution, state='readonly', width=14,
                     values=['640 x 480', '1280 x 720', '1920 x 1080']).pack(side='left')
        ttk.Button(row, text='Apply', command=self.reconnect).pack(side='left', padx=4)
        ttk.Label(sidebar, text='Bright light reduces camera noise. Preview is never enlarged.',
                  wraplength=350).pack(anchor='w', pady=5)
        ttk.Label(sidebar, textvariable=self.path_info, wraplength=350, foreground='#555555').pack(anchor='w')
        viewer = ttk.Frame(shell)
        viewer.pack(side='left', fill='both', expand=True)
        ttk.Label(viewer, textvariable=self.camera_info).pack(anchor='w')
        self.video = tk.Label(viewer, bg='#151b23', borderwidth=0)
        self.video.pack(fill='both', expand=True, pady=8)
        ttk.Label(viewer, textvariable=self.face_info, wraplength=650).pack(anchor='w')
        ttk.Label(viewer, textvariable=self.status, wraplength=650, font=self.status_font).pack(anchor='w', pady=8)
        ttk.Label(viewer, text='C one | Space burst | E extra | B extra burst | 1-5 subject | Q quit').pack(anchor='w')
        ttk.Label(viewer, text='Saved images contain the crop only. No display overlays or smoothing.').pack(anchor='w')

    @property
    def steps(self):
        return self.unknown_steps if self.person == '5' else self.sessions[self.session]['steps']

    @property
    def folder(self):
        sub = f'classmate_{self.classmate:02d}' if self.person == '5' else f's{self.session + 1}'
        return self.base / self.people[self.person] / sub

    def select_context(self):
        key = str(self.folder)
        if key not in self.stores:
            self.stores[key] = CaptureStore(self.folder, len(self.steps), self.shots)
        self.store = self.stores[key]
        self.pose = min(self.store.next_step, len(self.steps) - 1)
        self.frame = self.box = None  # never capture a frame from the previous subject
        self.auto.set(False)
        self.new_button.configure(state='normal' if self.person == '5' else 'disabled')
        if self.person == '5':
            self.heading.set(f'Unknown / classmate {self.classmate:02d}')
            self.environment.set('\n'.join(self.unknown_env))
        else:
            session = self.sessions[self.session]
            self.heading.set(f'Session {self.session + 1}/{len(self.sessions)}: {session["title"]}')
            self.environment.set('\n'.join(session['env']))
        self.path_info.set(f'Save folder: {self.folder}\nManual extras: extras/ subfolder')
        self.refresh()
        self.status.set('Ready. Old images use estimated pose counts; check them before continuing.'
                        if self.store.legacy else 'Ready. C captures one; E adds an extra photo at any time.')

    def refresh(self):
        counts = self.store.counts
        self.pose_combo.configure(values=[f'{i + 1}. {step} ({counts[i]}/{self.shots})'
                                           for i, step in enumerate(self.steps)])
        self.pose_combo.current(self.pose)
        done = sum(min(n, self.shots) for n in counts)
        target = len(counts) * self.shots
        self.progress.set(f'Guided: {done}/{target} | Extra photos: {self.store.extra_count}'
                          + ('\nGuided complete. Extra photos are still available.' if done >= target else ''))
        self.progress_bar['value'] = done / target
        self.ready_at = time.monotonic()

    def change_person(self, event=None):
        if self.remaining:
            self.member.set(f'{self.person}: {self.people[self.person]}')
            self.status.set('Stop the burst before changing subjects.')
            return
        person = self.member.get().split(':', 1)[0]
        classmate = max(classmate_indices(self.base / self.people['5']), default=1) if person == '5' else self.classmate
        self.switch_context(person=person, classmate=classmate)

    def switch_context(self, **changes):
        previous = self.person, self.session, self.classmate
        try:
            for name, value in changes.items():
                setattr(self, name, value)
            self.select_context()
        except (OSError, ValueError, KeyError, TypeError) as exc:
            self.person, self.session, self.classmate = previous
            self.member.set(f'{self.person}: {self.people[self.person]}')
            self.select_context()
            self.status.set(f'Cannot open that session: {exc}')

    def change_session(self, delta):
        if self.remaining:
            self.status.set('Stop the burst before changing sessions.')
            return
        if self.person == '5':
            # S creates the next session if needed; W opens the previous one.
            self.switch_context(classmate=max(1, self.classmate + delta))
        else:
            self.switch_context(session=max(0, min(len(self.sessions) - 1, self.session + delta)))

    def new_classmate(self):
        if self.person != '5':
            self.status.set('Select Unknown (5) before creating a new classmate session.')
            return
        if self.remaining:
            self.status.set('Stop the burst before starting a new classmate.')
            return
        self.switch_context(classmate=max(classmate_indices(self.base / self.people['5']), default=0) + 1)

    def change_pose(self, event=None):
        if self.remaining:
            self.pose_combo.current(self.pose)
            return
        self.auto.set(False)
        self.pose = self.pose_combo.current()
        self.refresh()

    def mode_changed(self, event=None):
        self.stop()
        self.frame = self.box = None
        self.status.set('Free manual saves unlimited extras.' if self.plan.get() == 'Free manual'
                        else 'Guided mode: C one photo, Space up to five, E extra photo.')

    def toggle_auto(self):
        if self.plan.get() == 'Free manual' or self.store.next_step >= len(self.steps) or self.remaining:
            self.auto.set(False)
            self.status.set('Auto requires guided mode and an unfinished pose. Use E for extras.')
            return
        if self.auto.get():
            self.pose = self.store.next_step
            self.refresh()
        self.ready_at = time.monotonic()

    def stop(self):
        saved = bool(self.remaining)
        self.remaining = 0
        self.auto.set(False)
        self.status.set('Stopped. Successfully saved photos are kept; resume the unfinished pose.'
                        if saved else 'Auto capture stopped.')

    def start_capture(self, count, extra=False, automatic=False):
        if self.remaining:
            return
        if not automatic:
            self.auto.set(False)
        now = time.monotonic()
        if now - self.last_save < 0.5:
            self.status.set('Please wait half a second before another capture.')
            return
        if self.frame is None or self.box is None:
            self.status.set('Show exactly one face, or select Centered square and align your face.')
            return
        extra = extra or self.plan.get() == 'Free manual'
        step = None if extra else self.pose
        if step is not None:
            count = min(count, self.shots - self.store.counts[step])
            if count <= 0:
                self.status.set('This pose is complete. Choose another pose or use E / Extra photo.')
                return
        self.store.begin_action()
        self.remaining, self.burst_step = count, step
        self.next_shot = now
        self.last_ok = now
        self.status.set(f'Capturing {count} photo(s)...')

    def undo(self):
        if self.remaining:
            self.status.set('Stop the burst before undoing it.')
            return
        self.auto.set(False)
        try:
            removed = self.store.undo()
            self.pose = min(self.store.next_step, len(self.steps) - 1)
            self.refresh()
            self.status.set(f'Undid {removed} photo(s) in this session.' if removed else 'Nothing to undo in this session.')
        except (OSError, ValueError) as exc:
            self.status.set(f'Undo failed: {exc}')

    def reconnect(self):
        self.stop()
        self.frame = self.box = None
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        try:
            index = int(self.camera_index.get())
            w, h = map(int, self.resolution.get().split(' x '))
            self.cap, frame = open_camera(index, w, h)
            self.camera_info.set(f'Camera {index} | Actual: {frame.shape[1]} x {frame.shape[0]} | Requested: {w} x {h}')
            self.failures = 0
            self.status.set('Camera connected. Keep your face inside the green square.')
        except (ValueError, RuntimeError, cv2.error) as exc:
            self.camera_info.set('Camera unavailable')
            self.status.set(str(exc))

    def _tick(self):
        if self.cap is None:
            return
        ok, raw = self.cap.read()
        if not ok or raw is None:
            self.frame = self.box = None
            self.failures += 1
            # Stop immediately; a failed camera must not leave a pending burst armed.
            self.stop()
            self.status.set('Camera read failed. Reconnect using Apply.')
            if self.failures >= 10:
                self.cap.release()
                self.cap = None
            return
        self.failures = 0
        self.frame = cv2.flip(raw, 1)
        now = time.monotonic()
        centered = self.crop_mode.get() == 'Centered square'
        faces = []
        if not centered:
            if self.detectors is None:
                self.detectors = load_detectors()
            faces = detect_faces(self.frame, *self.detectors)
        self.box = capture_box(self.frame, faces, centered)
        self.face_info.set('Centered crop: align one face inside the square.' if centered else
                           f'{len(faces)} face(s) detected. Capture requires exactly one face.')
        if self.remaining and now >= self.next_shot:
            if self.box is None:
                if now - self.last_ok > 2:
                    self.stop()
                    self.status.set('Face lost. Saved photos are kept. Resume this pose when ready.')
            else:
                crop = crop_image(self.frame, self.box, self.image_size)
                try:
                    path = self.store.save(crop, self.burst_step)
                except (OSError, ValueError, cv2.error) as exc:
                    self.stop()
                    self.status.set(f'Save failed: {exc}')
                else:
                    self.remaining -= 1
                    self.last_save = self.last_ok = now
                    self.next_shot = now + self.gap
                    self.refresh()
                    self.status.set(f'Saved {path.name}. {self.remaining} remaining.' if self.remaining else f'Saved: {path}')
                    print(f'Saved: {path}', flush=True)
                    if not self.remaining and self.burst_step is not None:
                        if self.store.counts[self.pose] >= self.shots:
                            self.pose = min(self.store.next_step, len(self.steps) - 1)
                            self.refresh()
                        if self.store.next_step >= len(self.steps):
                            self.auto.set(False)
                            self.status.set('Guided complete. E adds extras. S opens the next session; N starts a new Unknown classmate.')
        elif self.auto.get() and not self.remaining:
            if self.box is None:
                self.ready_at = now
                self.status.set('Auto paused: show exactly one face.')
            elif now - self.ready_at >= self.delay:
                self.start_capture(self.shots, automatic=True)
            else:
                self.status.set(f'Get into pose: capture in {max(1, int(self.delay - (now - self.ready_at)) + 1)}s')
        display = self.frame.copy()
        if self.box is not None:
            x, y, bw, bh = self.box
            cv2.rectangle(display, (x, y), (x + bw - 1, y + bh - 1), (70, 220, 110), 2)
        # Fit down only. Native widgets render the text independently of webcam pixels.
        h, w = display.shape[:2]
        scale = min(1.0, max(1, self.video.winfo_width()) / w, max(1, self.video.winfo_height()) / h)
        if scale < 1:
            display = cv2.resize(display, (max(1, round(w * scale)), max(1, round(h * scale))), interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(display, cv2.COLOR_BGR2RGB)
        dh, dw = rgb.shape[:2]
        self.photo = tk.PhotoImage(data=f'P6\n{dw} {dh}\n255\n'.encode('ascii') + rgb.tobytes(), format='PPM')
        self.video.configure(image=self.photo)

    def tick(self):
        if self.closed:
            return
        try:
            self._tick()
        except (cv2.error, RuntimeError, OSError, ValueError) as exc:
            self.frame = self.box = None
            self.stop()
            self.status.set(f'Capture error: {exc}. Try Centered square or reconnect the camera.')
        finally:
            if not self.closed:
                self.root.after(25, self.tick)

    def on_key(self, event):
        if sys.platform == 'darwin' and event.state & 0x0C:
            return  # Do not treat Command/Control shortcuts as capture keys.
        # Let text inputs and dropdowns keep their normal keyboard behavior.
        if isinstance(event.widget, (ttk.Combobox, ttk.Spinbox, ttk.Entry)):
            return
        key = event.keysym.lower()
        actions = {'c': lambda: self.start_capture(1), 'space': lambda: self.start_capture(self.shots),
                   'e': lambda: self.start_capture(1, extra=True),
                   'b': lambda: self.start_capture(self.shots, extra=True),
                   's': lambda: self.change_session(1), 'w': lambda: self.change_session(-1),
                   'n': self.new_classmate, 'u': self.undo, 'escape': self.stop, 'q': self.close}
        if key in self.people:
            self.member.set(f'{key}: {self.people[key]}')
            self.change_person()
        elif key in ('j', 'k') and not self.remaining:
            self.pose_combo.current(max(0, min(len(self.steps) - 1, self.pose + (1 if key == 'k' else -1))))
            self.change_pose()
        elif key == 'a':
            self.auto.set(not self.auto.get())
            self.toggle_auto()
        elif key in actions:
            actions[key]()
            return 'break'

    def close(self):
        if self.closed:
            return
        self.closed = True
        self.remaining = 0
        if self.cap is not None:
            self.cap.release()
        self.root.destroy()


def run(base, people, sessions, unknown_steps, unknown_env, *, image_size=(250, 250),
        shots=5, gap=0.2, delay=3.0, manual=False, camera_index=0):
    # Aware of Windows display scaling so native labels stay crisp on high-DPI screens.
    if os.name == 'nt':
        try:
            import ctypes
            ctypes.windll.shcore.SetProcessDpiAwareness(1)
        except (AttributeError, OSError):
            pass
    root = tk.Tk()
    app = None
    try:
        app = CollectorApp(root, base, people, sessions, unknown_steps, unknown_env,
                           image_size, shots, gap, delay, manual, camera_index=camera_index)
        root.mainloop()
    except (OSError, ValueError, KeyError, RuntimeError, cv2.error) as exc:
        messagebox.showerror('Dataset collector', str(exc), parent=root)
    finally:
        if app is not None and not app.closed:
            app.close()
        else:
            try:
                root.destroy()
            except tk.TclError:
                pass
