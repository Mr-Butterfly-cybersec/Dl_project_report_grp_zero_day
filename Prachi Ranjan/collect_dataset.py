"""Guided face dataset collector. Run collect_dataset_manual.py for free capture."""
from pathlib import Path

DATASET_DIR = Path(__file__).resolve().parent
IMAGE_SIZE = (250, 250)
SHOTS_PER_STEP = 5
BURST_GAP = 0.20
AUTO_DELAY = 3.0

PEOPLE = {
    "1": "Rahul",
    "2": "Prachi",
    "3": "Rishu",
    "4": "Gauri",
    "5": "Unknown",
}
UNKNOWN_KEY = "5"

# --------------------------------------------------------------
# SESSIONS: 6 sessions x 4 steps x 5 shots = 20 images / session
#           = 120 images per person
# Folder layout:  <script directory>/<Person>/s1 ... s6
# Split later:    s1-s4 -> train (80), s5 -> validate (20), s6 -> test (20)
# Tip: keep moving your head slightly DURING each burst so the 5 shots differ.
# --------------------------------------------------------------

SESSIONS = [
    {
        "title": "Plain wall | Normal light | Medium",
        "env": [
            "Stand in front of a PLAIN WALL",
            "Lights: normal classroom lights ON",
            "Distance: ~60 cm (arm's length)",
        ],
        "steps": [
            "Look STRAIGHT at camera (neutral)",
            "Look STRAIGHT and SMILE",
            "Turn head LEFT ~30 deg",
            "Turn head RIGHT ~30 deg",
        ],
    },
    {
        "title": "Whiteboard | Normal light | Close + Medium",
        "env": [
            "Stand in front of the WHITEBOARD/BLACKBOARD",
            "Lights: normal classroom lights ON",
            "Distance: CLOSE ~30 cm, then MEDIUM ~60 cm",
        ],
        "steps": [
            "CLOSE: straight, neutral",
            "MEDIUM: turn LEFT ~30 deg",
            "MEDIUM: turn RIGHT ~30 deg",
            "CLOSE: TALKING (mouth moving)",
        ],
    },
    {
        "title": "Window/corridor | Bright SIDE light | Medium",
        "env": [
            "Stand near a WINDOW or CORRIDOR",
            "Light must hit ONE SIDE of your face",
            "Distance: ~60 cm",
        ],
        "steps": [
            "Look STRAIGHT (neutral)",
            "Turn face TOWARD the light",
            "Turn face AWAY from the light",
            "SQUINT your eyes (bright light)",
        ],
    },
    {
        "title": "Plain wall | DIM light | Close + Medium",
        "env": [
            "Stand in front of a PLAIN WALL",
            "Lights: switch OFF most lights (DIM room)",
            "Distance: CLOSE ~30 cm, then MEDIUM ~60 cm",
        ],
        "steps": [
            "CLOSE: straight, neutral",
            "MEDIUM: straight, neutral",
            "MEDIUM: turn LEFT ~30 deg",
            "MEDIUM: turn RIGHT ~30 deg",
        ],
    },
    {
        "title": "Window BEHIND you | BACKLIT | Medium   (VALIDATE)",
        "env": [
            "Stand with the WINDOW BEHIND YOU",
            "Face the laptop, light comes from behind",
            "Distance: ~60 cm",
        ],
        "steps": [
            "Look STRAIGHT (neutral)",
            "Look STRAIGHT and SMILE",
            "Turn LEFT ~30 deg",
            "Turn RIGHT ~30 deg",
        ],
    },
    {
        "title": "Busy classroom | Normal light | Far + Medium   (TEST)",
        "env": [
            "Face the room - people/benches BEHIND you",
            "Lights: normal classroom lights ON",
            "Distance: FAR ~1.2 m, then MEDIUM ~60 cm",
        ],
        "steps": [
            "FAR: straight, neutral",
            "MEDIUM: straight, neutral",
            "FAR: turn LEFT ~30 deg",
            "FAR: SMILE",
        ],
    },
]

# Steps for the "Unknown" class (one classmate = 2 steps x 5 shots = 10 images)
UNKNOWN_ENV = [
    "Ask a CLASSMATE (get their consent first!)",
    "Any background / any light - vary per person",
    "Distance: ~60 cm. Press N for the next classmate",
]
UNKNOWN_STEPS = [
    "Classmate looks STRAIGHT at camera",
    "Classmate turns slightly LEFT or RIGHT (~30 deg)",
]


def main(manual=False):
    from collector_gui import run
    run(DATASET_DIR, PEOPLE, SESSIONS, UNKNOWN_STEPS, UNKNOWN_ENV,
        image_size=IMAGE_SIZE, shots=SHOTS_PER_STEP, gap=BURST_GAP,
        delay=AUTO_DELAY, manual=manual)


if __name__ == "__main__":
    main()
