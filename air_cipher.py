"""
Air-Cipher v4.0 — Dynamic Registration + Authentication
========================================================
Phase 1  REGISTRATION : Hold any supported gesture for 1.5 s, repeat
                        PASSCODE_LENGTH times to create your passcode.
Phase 2  AUTHENTICATION: Reproduce the exact sequence to unlock the system.

No passcode is hardcoded — everything is captured at runtime.

Supported gestures: Fist  |  1 Finger  |  2 Fingers  |  Thumbs Up
"""

import os
import subprocess
import sys
import time
import numpy as np
import cv2
import mediapipe as mp
from mediapipe.tasks import python as _mpp
from mediapipe.tasks.python import vision as _mpv
from datetime import datetime

# Detect the full display name of whoever is running the script (macOS)
try:
    _USER_NAME = subprocess.check_output(
        ["id", "-F"], stderr=subprocess.DEVNULL).decode().strip()
    if not _USER_NAME:
        raise ValueError
except Exception:
    _USER_NAME = os.getlogin()

# ══════════════════════════════════════════════════════════════════════════════
#  CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════
PASSCODE_LENGTH  = 3        # gestures to register / authenticate
SUSTAIN_SECONDS  = 1.5      # hold time to register or confirm a gesture
COOLDOWN_SECONDS = 0.5      # buffer between consecutive steps
CAMERA_ID        = 0
FRAME_W          = 1280
FRAME_H          = 720
MP_CONFIDENCE    = 0.70
HAND_MODEL_PATH  = "hand_landmarker.task"

# Add / remove gestures here.  All must be returnable by GestureRecognizer._classify().
KNOWN_GESTURES = frozenset({"Fist", "1 Finger", "2 Fingers", "Thumbs Up"})
# ══════════════════════════════════════════════════════════════════════════════

# ── Shared colours (BGR) ──────────────────────────────────────────────────────
C_DARK   = (14,  14,  18)
C_PANEL  = (22,  22,  28)
C_BORDER = (48,  48,  62)
C_GRAY   = (92,  92, 104)
C_WHITE  = (228, 228, 235)
C_GREEN  = (75,  215,  60)
C_RED    = (48,   48, 224)

# Authentication theme — amber / cyan
CA_GOLD  = (23,  160, 214)   # title & brackets
CA_CYAN  = (255, 207,   0)   # sustain arc
CA_COOL  = (255, 128,  58)   # cooldown

# Registration theme — violet
CR_VIO   = (220,  70, 185)   # title & brackets
CR_LAV   = (235, 130, 215)   # sustain arc
CR_COOL  = (240, 160, 100)   # cooldown

# ── Hand skeleton definition ──────────────────────────────────────────────────
_SKEL = {
    "palm":   {"color": (130, 130, 145),
               "pairs": [(0,1),(0,5),(0,17),(5,9),(9,13),(13,17)]},
    "thumb":  {"color": ( 23, 160, 214), "pairs": [(1,2),(2,3),(3,4)]},
    "index":  {"color": (255, 207,   0), "pairs": [(5,6),(6,7),(7,8)]},
    "middle": {"color": ( 75, 215,  60), "pairs": [(9,10),(10,11),(11,12)]},
    "ring":   {"color": (  0, 150, 255), "pairs": [(13,14),(14,15),(15,16)]},
    "pinky":  {"color": (200,  80, 200), "pairs": [(17,18),(18,19),(19,20)]},
}
_TIPS = {4, 8, 12, 16, 20}


# ─────────────────────────────────────────────────────────────────────────────
class GestureRecognizer:
    """MediaPipe HandLandmarker wrapper (Tasks API, mediapipe ≥ 0.10.21)."""

    def __init__(self):
        base = _mpp.BaseOptions(model_asset_path=HAND_MODEL_PATH)
        opts = _mpv.HandLandmarkerOptions(
            base_options=base,
            running_mode=_mpv.RunningMode.IMAGE,
            num_hands=1,
            min_hand_detection_confidence=MP_CONFIDENCE,
            min_hand_presence_confidence=MP_CONFIDENCE,
            min_tracking_confidence=MP_CONFIDENCE,
        )
        self._det = _mpv.HandLandmarker.create_from_options(opts)

    def process(self, bgr_frame: np.ndarray) -> tuple:
        """Returns (gesture: str, landmark_list | None)."""
        rgb    = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self._det.detect(mp_img)
        if not result.hand_landmarks:
            return "Background", None
        lm = result.hand_landmarks[0]
        return self._classify(lm), lm

    @staticmethod
    def _classify(lm) -> str:
        """
        Classify from 21 NormalizedLandmarks.
        Y is image-space (increases downward): TIP.y < PIP.y → finger extended.
        Thumb-up: tip above its own MCP and above the index-finger MCP.
        """
        i_up = lm[8].y  < lm[6].y
        m_up = lm[12].y < lm[10].y
        r_up = lm[16].y < lm[14].y
        p_up = lm[20].y < lm[18].y
        t_up = lm[4].y  < lm[2].y and lm[4].y < lm[5].y
        n    = i_up + m_up + r_up + p_up
        if n == 0:
            return "Thumbs Up" if t_up else "Fist"
        if n == 1 and i_up:
            return "1 Finger"
        if n == 2 and i_up and m_up:
            return "2 Fingers"
        return "Other"

    def close(self):
        self._det.close()


# ─────────────────────────────────────────────────────────────────────────────
class RegistrationMachine:
    """
    Captures PASSCODE_LENGTH gestures to build the user's passcode.

    States
    ──────
    WAITING  — waiting for user to show any known gesture
    COOLDOWN — 0.5 s buffer after capturing a gesture
    COMPLETE — all gestures captured; passcode is ready
    """

    WAITING  = "WAITING"
    COOLDOWN = "COOLDOWN"
    COMPLETE = "COMPLETE"

    def __init__(self, sustain_s: float = SUSTAIN_SECONDS,
                 cooldown_s: float = COOLDOWN_SECONDS):
        self._sus_s   = sustain_s
        self._cool_s  = cooldown_s
        self.step     = 0
        self.status   = self.WAITING
        self.captured : list[str] = []
        self._last_g  = ""
        self._sus_t   = None
        self._cool_t  = None

    def update(self, gesture: str) -> dict:
        now = time.time()
        g   = gesture if gesture in KNOWN_GESTURES else "Background"

        if self.status == self.COMPLETE:
            return self._snap(g, 1.0, 0.0)

        if self.status == self.COOLDOWN:
            elapsed = now - self._cool_t
            cd      = min(elapsed / self._cool_s, 1.0)
            if elapsed >= self._cool_s:
                self.status  = self.WAITING
                self._sus_t  = None
                self._last_g = ""
            return self._snap(g, 0.0, cd)

        # WAITING — run sustain timer
        if g != self._last_g:
            self._last_g = g
            self._sus_t  = now if g != "Background" else None

        if g == "Background" or self._sus_t is None:
            self._sus_t = None
            return self._snap(g, 0.0, 0.0)

        progress = min((now - self._sus_t) / self._sus_s, 1.0)

        if progress >= 1.0:
            self.captured.append(g)
            self.step += 1
            if self.step == PASSCODE_LENGTH:
                self.status = self.COMPLETE
            else:
                self.status  = self.COOLDOWN
                self._cool_t = now
                self._last_g = ""

        return self._snap(g, progress, 0.0)

    def _snap(self, g, prog, cd) -> dict:
        return {"status": self.status, "step": self.step,
                "progress": prog, "cd_progress": cd,
                "gesture": g, "captured": list(self.captured)}


# ─────────────────────────────────────────────────────────────────────────────
class AuthMachine:
    """
    Verifies the user's live input against the registered passcode.

    States
    ──────
    IDLE     — waiting for the next gesture in the sequence
    COOLDOWN — 0.5 s buffer after a correct step
    SUCCESS  — full passcode matched; system unlocked
    FAILED   — wrong gesture held; resets after 1.8 s
    """

    IDLE     = "IDLE"
    COOLDOWN = "COOLDOWN"
    SUCCESS  = "SUCCESS"
    FAILED   = "FAILED"

    def __init__(self, passcode: list[str],
                 sustain_s: float = SUSTAIN_SECONDS,
                 cooldown_s: float = COOLDOWN_SECONDS):
        self.passcode = passcode
        self._sus_s   = sustain_s
        self._cool_s  = cooldown_s
        self._reset()

    def _reset(self):
        self.step      = 0
        self.status    = self.IDLE
        self.confirmed : list[bool] = []
        self._last_g   = ""
        self._sus_t    = None
        self._cool_t   = None
        self._fail_t   = None

    def update(self, gesture: str) -> dict:
        now = time.time()
        g   = gesture if gesture in KNOWN_GESTURES else "Background"

        if self.status == self.FAILED:
            if now - self._fail_t >= 1.8:
                self._reset()
            return self._snap(g, 0.0, 0.0)

        if self.status == self.SUCCESS:
            return self._snap(g, 1.0, 0.0)

        if self.status == self.COOLDOWN:
            elapsed = now - self._cool_t
            cd      = min(elapsed / self._cool_s, 1.0)
            if elapsed >= self._cool_s:
                self.status  = self.IDLE
                self._sus_t  = None
                self._last_g = ""
            return self._snap(g, 0.0, cd)

        # IDLE
        if g != self._last_g:
            self._last_g = g
            self._sus_t  = now if g != "Background" else None

        if g == "Background" or self._sus_t is None:
            self._sus_t = None
            return self._snap(g, 0.0, 0.0)

        progress = min((now - self._sus_t) / self._sus_s, 1.0)

        if progress >= 1.0:
            if g == self.passcode[self.step]:
                self.confirmed.append(True)
                self.step += 1
                if self.step == len(self.passcode):
                    self.status = self.SUCCESS
                    _trigger_success()
                else:
                    self.status  = self.COOLDOWN
                    self._cool_t = now
                    self._last_g = ""
            else:
                self.confirmed.append(False)
                self.status  = self.FAILED
                self._fail_t = now

        return self._snap(g, progress, 0.0)

    def _snap(self, g, prog, cd) -> dict:
        return {"status": self.status, "step": self.step,
                "progress": prog, "cd_progress": cd,
                "gesture": g, "confirmed": list(self.confirmed)}


# ─────────────────────────────────────────────────────────────────────────────
class UIRenderer:
    """
    Phase-aware HUD renderer.
    Registration → violet theme.
    Transition   → green confirmation screen.
    Authentication → gold theme with lock overlay.
    """

    _F  = cv2.FONT_HERSHEY_SIMPLEX
    _FM = cv2.FONT_HERSHEY_DUPLEX

    # ── Phase entry points ────────────────────────────────────────────────────

    def draw_registration(self, frame: np.ndarray, state: dict,
                          landmarks, fps: float) -> np.ndarray:
        h, w = frame.shape[:2]
        ov   = frame.copy()
        if landmarks is not None:
            self._skeleton(ov, landmarks, w, h,
                           state["progress"], state["status"],
                           phase="REG")
        self._top_bar(ov, w, fps, "REGISTRATION MODE", CR_VIO)
        self._brackets(ov, h, w, CR_VIO)
        self._gesture_panel(ov, h, state["gesture"], state["progress"],
                            CR_VIO, CR_LAV)
        self._reg_boxes(ov, h, w, state)
        self._reg_hint(ov, h, w, state)
        if state["status"] == RegistrationMachine.COOLDOWN:
            self._cd_bar(ov, h, w, state["cd_progress"], CR_COOL)
        cv2.addWeighted(ov, 0.88, frame, 0.12, 0, frame)
        return frame

    def draw_transition(self, frame: np.ndarray, captured: list[str],
                        fps: float) -> np.ndarray:
        h, w = frame.shape[:2]
        ov   = frame.copy()
        # Dim the live camera feed
        dark = np.full_like(ov, (10, 10, 14))
        cv2.addWeighted(dark, 0.65, ov, 0.35, 0, ov)

        self._top_bar(ov, w, fps, "SYSTEM PREPARING", C_GREEN)

        msg = "PASSCODE REGISTERED"
        (mw, mh), _ = cv2.getTextSize(msg, self._FM, 1.4, 2)
        cv2.putText(ov, msg, ((w - mw) // 2, h // 2 - 70),
                    self._FM, 1.4, C_GREEN, 2, cv2.LINE_AA)

        sub = "Entering Lock Mode..."
        (sw, _), _ = cv2.getTextSize(sub, self._F, 0.65, 1)
        cv2.putText(ov, sub, ((w - sw) // 2, h // 2 - 18),
                    self._F, 0.65, C_GRAY, 1, cv2.LINE_AA)

        # Show the captured gesture sequence
        for k, g in enumerate(captured):
            line = f"{k + 1}.  {g}"
            (lw, _), _ = cv2.getTextSize(line, self._FM, 0.72, 1)
            cv2.putText(ov, line, ((w - lw) // 2, h // 2 + 44 + k * 44),
                        self._FM, 0.72, C_GREEN, 1, cv2.LINE_AA)

        cv2.addWeighted(ov, 0.95, frame, 0.05, 0, frame)
        return frame

    def draw_auth(self, frame: np.ndarray, state: dict,
                  landmarks, fps: float) -> np.ndarray:
        h, w = frame.shape[:2]
        ov   = frame.copy()
        if landmarks is not None:
            self._skeleton(ov, landmarks, w, h,
                           state["progress"], state["status"],
                           phase="AUTH")
        # Locked overlay — only shown when completely idle at step 0
        if (state["status"] == AuthMachine.IDLE
                and state["step"] == 0
                and state["progress"] == 0.0):
            self._locked_overlay(ov, h, w)
        self._top_bar(ov, w, fps, "AUTHENTICATION MODE", CA_GOLD)
        self._brackets(ov, h, w, CA_GOLD)
        self._gesture_panel(ov, h, state["gesture"], state["progress"],
                            CA_GOLD, CA_CYAN)
        self._auth_boxes(ov, h, w, state)
        self._auth_hint(ov, h, w, state)
        if state["status"] == AuthMachine.COOLDOWN:
            self._cd_bar(ov, h, w, state["cd_progress"], CA_COOL)
        self._auth_banner(ov, h, w, state)
        cv2.addWeighted(ov, 0.88, frame, 0.12, 0, frame)
        return frame

    # ── Locked overlay ────────────────────────────────────────────────────────

    def _locked_overlay(self, img, h, w):
        cv2.rectangle(img, (3, 70), (w - 3, h - 3), C_RED, 4)
        self._panel(img, 0, 70, w, 112, (32, 10, 20), 0.82)
        msg = "[  LOCKED  ]   SHOW YOUR GESTURE PASSCODE TO UNLOCK"
        (mw, _), _ = cv2.getTextSize(msg, self._F, 0.52, 1)
        cv2.putText(img, msg, ((w - mw) // 2, 98),
                    self._F, 0.52, C_RED, 1, cv2.LINE_AA)

    # ── Hand skeleton ─────────────────────────────────────────────────────────

    def _skeleton(self, img, lm, w, h, progress, status, phase):
        pts = [(int(l.x * w), int(l.y * h)) for l in lm]

        if phase == "AUTH":
            if status == AuthMachine.SUCCESS:
                gc, gt = C_GREEN, 0.90
            elif status == AuthMachine.FAILED:
                gc, gt = C_RED,   0.82
            elif progress > 0:
                gc, gt = CA_CYAN, progress * 0.65
            else:
                gc, gt = None, 0.0
        else:
            if status == RegistrationMachine.COMPLETE:
                gc, gt = C_GREEN, 0.85
            elif progress > 0:
                gc, gt = CR_LAV, progress * 0.65
            else:
                gc, gt = None, 0.0

        for part, spec in _SKEL.items():
            base = spec["color"]
            col  = self._lerp(base, gc, gt) if gc else base
            for a, b in spec["pairs"]:
                cv2.line(img, pts[a], pts[b], self._dim(col, 0.32), 9,  cv2.LINE_AA)
                cv2.line(img, pts[a], pts[b], col,                  3,  cv2.LINE_AA)

        for i, pt in enumerate(pts):
            r = 7 if i in _TIPS else 4
            cv2.circle(img, pt, r + 2, C_DARK,  -1)
            cv2.circle(img, pt, r,     C_WHITE, -1)

    # ── Registration boxes ────────────────────────────────────────────────────

    def _reg_boxes(self, img, h, w, state):
        sz, gap = 122, 20
        tw  = PASSCODE_LENGTH * sz + (PASSCODE_LENGTH - 1) * gap
        x0  = (w - tw) // 2
        y0  = h - sz - 62

        captured = state["captured"]
        step     = state["step"]
        status   = state["status"]

        for i in range(PASSCODE_LENGTH):
            bx = x0 + i * (sz + gap)
            by = y0
            cx = bx + sz // 2
            cy = by + sz // 2 - 8

            done      = i < len(captured)
            is_active = (i == step and status not in
                         (RegistrationMachine.COOLDOWN,
                          RegistrationMachine.COMPLETE))

            if done:
                bg, bdr, tc = (20, 50, 22), C_GREEN, C_GREEN
            elif is_active:
                bg, bdr, tc = (28, 18, 36), CR_VIO, CR_VIO
            else:
                bg, bdr, tc = C_PANEL, C_BORDER, C_GRAY

            self._panel(img, bx, by, bx + sz, by + sz, bg, 0.90)
            cv2.rectangle(img, (bx, by), (bx + sz, by + sz), bdr, 2)
            cv2.putText(img, str(i + 1), (bx + 8, by + 20),
                        self._F, 0.40, tc, 1, cv2.LINE_AA)

            r = 34
            if done:
                lbl   = captured[i].upper()
                sc    = 0.36 if len(lbl) > 7 else 0.40
                (lw, _), _ = cv2.getTextSize(lbl, self._F, sc, 1)
                cv2.putText(img, lbl, (bx + (sz - lw) // 2, by + sz - 10),
                            self._F, sc, C_GREEN, 1, cv2.LINE_AA)
                (ow, oh), _ = cv2.getTextSize("OK", self._FM, 1.0, 2)
                cv2.putText(img, "OK", (cx - ow // 2, cy + oh // 2),
                            self._FM, 1.0, C_GREEN, 2, cv2.LINE_AA)
            elif is_active:
                self._arc(img, cx, cy, r, 0, 360, C_BORDER, 5)
                ang = int(360 * state["progress"])
                if ang > 0:
                    self._arc(img, cx, cy, r, 0, ang, CR_LAV, 5)
                pct = f"{int(state['progress'] * 100)}%"
                (pw, ph), _ = cv2.getTextSize(pct, self._F, 0.42, 1)
                cv2.putText(img, pct, (cx - pw // 2, cy + ph // 2),
                            self._F, 0.42, CR_VIO, 1, cv2.LINE_AA)
            else:
                self._arc(img, cx, cy, r, 0, 360, C_BORDER, 3)
                cv2.putText(img, "?", (cx - 8, cy + 10),
                            self._FM, 0.85, C_BORDER, 2, cv2.LINE_AA)

    # ── Auth boxes ────────────────────────────────────────────────────────────

    def _auth_boxes(self, img, h, w, state):
        sz, gap = 122, 20
        tw  = PASSCODE_LENGTH * sz + (PASSCODE_LENGTH - 1) * gap
        x0  = (w - tw) // 2
        y0  = h - sz - 62

        confirmed = state["confirmed"]
        step      = state["step"]
        status    = state["status"]
        AM        = AuthMachine

        for i in range(PASSCODE_LENGTH):
            bx = x0 + i * (sz + gap)
            by = y0
            cx = bx + sz // 2
            cy = by + sz // 2 - 8

            done      = i < len(confirmed)
            ok        = confirmed[i] if done else False
            is_active = (i == step and status not in
                         (AM.COOLDOWN, AM.SUCCESS, AM.FAILED))

            if done and ok:
                bg, bdr, tc = (18, 52, 22), C_GREEN, C_GREEN
            elif done and not ok:
                bg, bdr, tc = (40, 15, 18), C_RED,   C_RED
            elif is_active:
                bg, bdr, tc = (18, 22, 32), CA_CYAN, CA_CYAN
            else:
                bg, bdr, tc = C_PANEL, C_BORDER, C_GRAY

            self._panel(img, bx, by, bx + sz, by + sz, bg, 0.90)
            cv2.rectangle(img, (bx, by), (bx + sz, by + sz), bdr, 2)
            cv2.putText(img, str(i + 1), (bx + 8, by + 20),
                        self._F, 0.40, tc, 1, cv2.LINE_AA)

            r = 34
            if done:
                mark = "OK" if ok else "X"
                mc   = C_GREEN if ok else C_RED
                (mw, mh), _ = cv2.getTextSize(mark, self._FM, 1.05, 2)
                cv2.putText(img, mark, (cx - mw // 2, cy + mh // 2),
                            self._FM, 1.05, mc, 2, cv2.LINE_AA)
            elif is_active:
                self._arc(img, cx, cy, r, 0, 360, C_BORDER, 5)
                ang = int(360 * state["progress"])
                if ang > 0:
                    self._arc(img, cx, cy, r, 0, ang, CA_CYAN, 5)
                pct = f"{int(state['progress'] * 100)}%"
                (pw, ph), _ = cv2.getTextSize(pct, self._F, 0.42, 1)
                cv2.putText(img, pct, (cx - pw // 2, cy + ph // 2),
                            self._F, 0.42, CA_CYAN, 1, cv2.LINE_AA)
            else:
                self._arc(img, cx, cy, r, 0, 360, C_BORDER, 3)
                cv2.putText(img, "?", (cx - 8, cy + 10),
                            self._FM, 0.85, C_BORDER, 2, cv2.LINE_AA)

    # ── Hints ─────────────────────────────────────────────────────────────────

    def _reg_hint(self, img, h, w, state):
        status = state["status"]
        step   = state["step"]
        if status == RegistrationMachine.COMPLETE:
            text, col = "ALL GESTURES CAPTURED — SAVING PASSCODE...", C_GREEN
        elif status == RegistrationMachine.COOLDOWN:
            text = f"GESTURE {step} OF {PASSCODE_LENGTH} CAPTURED — RELAX YOUR HAND..."
            col  = CR_VIO
        else:
            text = (f"REGISTER GESTURE  {step + 1}  OF  {PASSCODE_LENGTH}"
                    f"  —  HOLD ANY GESTURE FOR {SUSTAIN_SECONDS:.0f}s")
            col  = CR_VIO
        (tw, _), _ = cv2.getTextSize(text, self._F, 0.46, 1)
        cv2.putText(img, text, ((w - tw) // 2, h - 192),
                    self._F, 0.46, col, 1, cv2.LINE_AA)

    def _auth_hint(self, img, h, w, state):
        status = state["status"]
        step   = state["step"]
        AM     = AuthMachine
        if status in (AM.SUCCESS, AM.FAILED):
            return
        if status == AM.COOLDOWN:
            text, col = "CHANGING STEP — RELAX YOUR HAND...", CA_COOL
        elif step < PASSCODE_LENGTH:
            text = f"SHOW GESTURE  {step + 1}  OF  {PASSCODE_LENGTH}"
            col  = CA_GOLD
        else:
            return
        (tw, _), _ = cv2.getTextSize(text, self._F, 0.48, 1)
        cv2.putText(img, text, ((w - tw) // 2, h - 192),
                    self._F, 0.48, col, 1, cv2.LINE_AA)

    # ── Shared elements ───────────────────────────────────────────────────────

    def _top_bar(self, img, w, fps, subtitle, accent):
        self._panel(img, 0, 0, w, 66, C_DARK, 0.94)
        cv2.line(img, (0, 64), (w, 64), accent, 2)
        cv2.putText(img, "AIR-CIPHER", (20, 44),
                    self._FM, 1.0, accent, 1, cv2.LINE_AA)
        (sw, _), _ = cv2.getTextSize(subtitle, self._F, 0.45, 1)
        cv2.putText(img, subtitle, ((w - sw) // 2, 44),
                    self._F, 0.45, accent, 1, cv2.LINE_AA)
        right = f"{datetime.now():%H:%M:%S}   FPS {fps:4.0f}"
        (rw, _), _ = cv2.getTextSize(right, self._F, 0.44, 1)
        cv2.putText(img, right, (w - rw - 20, 44),
                    self._F, 0.44, C_GRAY, 1, cv2.LINE_AA)

    def _brackets(self, img, h, w, color):
        m, L, t = 10, 50, 3
        for cx, cy, dx, dy in [
            (m,     70 + m,  1,  1),
            (w - m, 70 + m, -1,  1),
            (m,     h - m,   1, -1),
            (w - m, h - m,  -1, -1),
        ]:
            cv2.line(img, (cx, cy), (cx + dx * L, cy), color, t, cv2.LINE_AA)
            cv2.line(img, (cx, cy), (cx, cy + dy * L), color, t, cv2.LINE_AA)

    def _gesture_panel(self, img, h, gesture, progress, accent, arc_col):
        px, py, pw = 14, h - 150, 262
        self._panel(img, px, py, px + pw, py + 130, C_PANEL, 0.86)
        cv2.rectangle(img, (px, py), (px + pw, py + 130), C_BORDER, 1)
        cv2.line(img, (px, py + 26), (px + pw, py + 26), C_BORDER, 1)
        cv2.putText(img, "DETECTED GESTURE", (px + 10, py + 18),
                    self._F, 0.36, C_GRAY, 1, cv2.LINE_AA)
        active = gesture not in ("Background", "Other")
        col    = C_GREEN if active else C_GRAY
        scale  = 0.80 if len(gesture) <= 7 else 0.64
        cv2.putText(img, gesture.upper(), (px + 10, py + 66),
                    self._FM, scale, col, 1, cv2.LINE_AA)
        bx, bw = px + 10, pw - 20
        by_b   = py + 82
        cv2.rectangle(img, (bx, by_b), (bx + bw, by_b + 10), C_BORDER, -1)
        f = int(bw * progress)
        if f > 0:
            cv2.rectangle(img, (bx, by_b), (bx + f, by_b + 10),
                          C_GREEN if progress >= 1.0 else arc_col, -1)
        pct = f"{progress * 100:.0f}%"
        (pw2, _), _ = cv2.getTextSize(pct, self._F, 0.36, 1)
        cv2.putText(img, pct, (bx + bw - pw2, by_b + 24),
                    self._F, 0.36, C_GRAY, 1, cv2.LINE_AA)

    def _cd_bar(self, img, h, w, cd_progress, color):
        sz, gap = 122, 20
        bw = PASSCODE_LENGTH * sz + (PASSCODE_LENGTH - 1) * gap
        bx = (w - bw) // 2
        by = h - 62 + 4
        cv2.rectangle(img, (bx, by), (bx + bw, by + 6), C_BORDER, -1)
        f = int(bw * cd_progress)
        if f > 0:
            cv2.rectangle(img, (bx, by), (bx + f, by + 6), color, -1)
        lbl = "NEXT STEP READY"
        (lw, _), _ = cv2.getTextSize(lbl, self._F, 0.36, 1)
        cv2.putText(img, lbl, ((w - lw) // 2, by + 20),
                    self._F, 0.36, color, 1, cv2.LINE_AA)

    def _auth_banner(self, img, h, w, state):
        AM = AuthMachine
        if state["status"] == AM.SUCCESS:
            lines = ["ACCESS GRANTED",
                     f"SUCCESS: SYSTEM UNLOCKED — WELCOME, {_USER_NAME.upper()}"]
            col   = C_GREEN
        elif state["status"] == AM.FAILED:
            lines = ["WRONG GESTURE", "RESETTING SEQUENCE..."]
            col   = C_RED
        else:
            return

        sizes  = [cv2.getTextSize(l, self._FM, 1.05, 2)[0] for l in lines]
        max_w  = max(s[0] for s in sizes)
        line_h = sizes[0][1]
        pad    = 26
        tot_h  = len(lines) * (line_h + 14)
        bx = (w - max_w) // 2 - pad
        by = h // 2 - tot_h // 2 - pad
        bw = max_w + 2 * pad
        bh = tot_h + 2 * pad + 14

        self._panel(img, bx, by, bx + bw, by + bh, C_DARK, 0.93)
        cv2.rectangle(img, (bx, by), (bx + bw, by + bh), col, 2)
        cv2.rectangle(img, (bx+4, by+4), (bx+bw-4, by+bh-4), col, 1)

        for k, (line, (lw, lh)) in enumerate(zip(lines, sizes)):
            cv2.putText(img, line, ((w - lw) // 2, by + pad + lh + k * (lh + 14)),
                        self._FM, 1.05, col, 2, cv2.LINE_AA)

    # ── Low-level helpers ─────────────────────────────────────────────────────

    @staticmethod
    def _lerp(c1, c2, t):
        return tuple(int(c1[i] + (c2[i] - c1[i]) * t) for i in range(3))

    @staticmethod
    def _dim(c, f):
        return tuple(int(x * f) for x in c)

    @staticmethod
    def _panel(img, x0, y0, x1, y1, color=C_PANEL, alpha=0.82):
        roi  = img[y0:y1, x0:x1]
        fill = np.full_like(roi, color)
        cv2.addWeighted(fill, alpha, roi, 1 - alpha, 0, roi)
        img[y0:y1, x0:x1] = roi

    @staticmethod
    def _arc(img, cx, cy, r, a0, a1, color, thick=5):
        cv2.ellipse(img, (cx, cy), (r, r),
                    -90, a0, a1, color, thick, cv2.LINE_AA)


# ─────────────────────────────────────────────────────────────────────────────
def _trigger_success():
    """Clear the macOS terminal and print the secure-access banner."""
    print("\033[2J\033[H", end="", flush=True)
    b   = "═" * 64
    msg = f"ACCESS GRANTED TO {_USER_NAME.upper()}'S ENVIRONMENT"
    print(f"╔{b}╗")
    print(f"║{'SUCCESS: SYSTEM UNLOCKED':^64}║")
    print(f"║{msg:^64}║")
    print(f"╚{b}╝")


# ─────────────────────────────────────────────────────────────────────────────
def main():
    recognizer  = GestureRecognizer()
    renderer    = UIRenderer()
    reg         = RegistrationMachine()
    auth        = None
    phase       = "REGISTRATION"   # REGISTRATION → TRANSITION → AUTHENTICATION
    trans_start = None

    cap = cv2.VideoCapture(CAMERA_ID)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  FRAME_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)
    if not cap.isOpened():
        sys.exit("[ERROR] Cannot open webcam.")

    prev_t = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame  = cv2.flip(frame, 1)
        gesture, landmarks = recognizer.process(frame)

        now    = time.time()
        fps    = 1.0 / max(now - prev_t, 1e-6)
        prev_t = now

        if phase == "REGISTRATION":
            state = reg.update(gesture)
            frame = renderer.draw_registration(frame, state, landmarks, fps)
            if state["status"] == RegistrationMachine.COMPLETE:
                auth        = AuthMachine(list(reg.captured))
                phase       = "TRANSITION"
                trans_start = now

        elif phase == "TRANSITION":
            frame = renderer.draw_transition(frame, reg.captured, fps)
            if now - trans_start >= 2.5:
                phase = "AUTHENTICATION"

        elif phase == "AUTHENTICATION":
            state = auth.update(gesture)
            frame = renderer.draw_auth(frame, state, landmarks, fps)

        cv2.imshow("Air-Cipher", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    recognizer.close()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
