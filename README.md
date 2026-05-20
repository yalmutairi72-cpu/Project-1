# Air-Cipher — Sequential Gesture Authenticator

A real-time, contactless security system that uses **hand gesture recognition** to authenticate users through a secret sequence of gestures — no passwords, no typing, no physical contact.

Built with **MediaPipe Hands** and **OpenCV**. No machine learning model training required.

---

## Demo

| Phase | What you see |
|---|---|
| **Registration** | Violet UI — hold any 3 gestures for 1.5s each to create your passcode |
| **Transition** | Green confirmation screen showing the captured sequence |
| **Authentication** | Gold UI — reproduce your exact sequence to unlock the system |
| **Success** | Terminal clears and prints a personalised access banner |

---

## How It Works

### Phase 1 — Registration
When the app launches, it enters **Registration Mode** (violet theme).  
The user holds any supported gesture steady for **1.5 seconds** — the animated arc fills up as a sustain timer.  
Once registered, a **0.5-second cooldown** gives the user time to naturally switch to the next gesture.  
This repeats until all 3 gestures are captured. The passcode is stored in memory — nothing is hardcoded or saved to disk.

### Phase 2 — Authentication
The system immediately enters **Lock Mode** (gold theme) with a red border overlay.  
The user must reproduce the **exact same sequence** they just registered — same gestures, same order.  
Each step uses the same 1.5s sustain mechanic. A wrong gesture triggers a **FAILED** state and resets the sequence.

### On Success
- The terminal clears and prints a personalised banner with the **name of the current system user** (detected automatically from the OS — works on any Mac).
- The on-screen overlay displays `ACCESS GRANTED — WELCOME, [YOUR NAME]`.

---

## Gesture Library

| Gesture | How to perform it |
|---|---|
| **1 Finger** | Point index finger up, all others closed |
| **2 Fingers** | Index and middle fingers up (peace sign) |
| **Fist** | All fingers closed tightly |
| **Thumbs Up** | Thumb pointing up, all other fingers closed |

Detection is **lighting-independent** — it uses the geometric positions of 21 hand landmarks, not pixel colors or textures.

---

## Gesture Detection — How the Maths Works

MediaPipe tracks **21 landmarks** on the hand in normalised image coordinates (x, y).  
Since image Y increases **downward**, a finger is considered *extended* when its **tip Y < PIP Y** (the tip is higher on screen than the middle knuckle).

```
Finger UP   →  TIP.y  <  PIP.y
Finger DOWN →  TIP.y  >  PIP.y

Thumbs Up   →  all 4 fingers DOWN
               AND  thumb_tip.y < thumb_MCP.y
               AND  thumb_tip.y < index_MCP.y
```

---

## State Machine

```
REGISTRATION
    WAITING ──(sustain 1.5s)──► COOLDOWN ──(0.5s)──► WAITING
                                                        │
                                              (×PASSCODE_LENGTH)
                                                        │
                                                     COMPLETE
                                                        │
                                             ┌──────────▼──────────┐
                                             │  2.5s splash screen │
                                             └──────────┬──────────┘
AUTHENTICATION                                          │
    IDLE ──(sustain 1.5s)──► correct ──► COOLDOWN ──► IDLE
                          └─► wrong  ──► FAILED (auto-reset 1.8s)
                                                        │
                                            (all steps correct)
                                                        │
                                                     SUCCESS
```

---

## Installation

**Requirements:** Python 3.10 – 3.12, macOS (Apple Silicon or Intel)

```bash
# Clone the repository
git clone https://github.com/yalmutairi72-cpu/Project-1.git
cd Project-1

# Create a virtual environment with Python 3.12
python3.12 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

> **Apple Silicon (M1/M2/M3):** All packages have native arm64 wheels — no extra steps needed.

---

## Running the App

```bash
source .venv/bin/activate
python air_cipher.py
```

On first launch, grant **camera access** to your terminal app when macOS prompts you  
(*System Settings → Privacy & Security → Camera*).

Press **`q`** to quit at any time.

---

## Configuration

All settings are at the top of `air_cipher.py`:

```python
PASSCODE_LENGTH  = 3      # number of gestures to register
SUSTAIN_SECONDS  = 1.5    # hold time to register / confirm a gesture
COOLDOWN_SECONDS = 0.5    # buffer between steps
MP_CONFIDENCE    = 0.70   # MediaPipe detection confidence (0–1)
```

To add more gesture types, extend the `_classify()` method inside `GestureRecognizer` and add the new name to `KNOWN_GESTURES`.

---

## Project Structure

```
Project-1/
├── air_cipher.py          # Main application
├── hand_landmarker.task   # MediaPipe hand landmark model (7.8 MB)
├── requirements.txt       # Python dependencies
├── create_dummy_model.py  # Legacy helper — generates a dummy .h5 model
├── keras_model.h5         # Legacy dummy TensorFlow model (not used)
└── labels.txt             # Legacy labels file (not used)
```

---

## Dependencies

```
mediapipe>=0.10.0
opencv-python>=4.8.0
numpy>=1.24.0
```

No TensorFlow. No training data. No model files to manage (the MediaPipe model is bundled).

---

## Security Notes

- The registered passcode exists **only in RAM** for the duration of the session. It is never written to disk.
- The system username is read from the OS at runtime using `id -F` — it is never transmitted anywhere.
- This project is a proof-of-concept built for a hackathon demo and is not intended for production security use.

---

## Built With

- [MediaPipe](https://developers.google.com/mediapipe) — hand landmark detection
- [OpenCV](https://opencv.org/) — camera capture and UI rendering
- [NumPy](https://numpy.org/) — array operations

---

## Author

**Yousef Almutairi**  
Built for an AI Hackathon — 2025
