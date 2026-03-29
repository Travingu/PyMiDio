import mido
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel, QSizePolicy
from PyQt6.QtCore import Qt, QTimer, QRectF, pyqtSignal
from PyQt6.QtGui import QPainter, QColor, QPen, QFont, QBrush, QLinearGradient

NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
BLACK_KEYS = {1, 3, 6, 8, 10}

ROLL_BG        = QColor(18, 18, 24)
LANE_ALT       = QColor(24, 24, 32)
GRID_LINE      = QColor(40, 40, 55)
GRID_BEAT      = QColor(55, 55, 75)
NOTE_COLOR     = QColor(42, 130, 218)
NOTE_HIGHLIGHT = QColor(80, 180, 255)
NOTE_BORDER    = QColor(100, 200, 255)
BLACK_KEY_NOTE = QColor(28, 90, 160)
PLAYHEAD_COLOR = QColor(255, 80, 80)
KEY_WHITE_BG   = QColor(220, 220, 225)
KEY_BLACK_BG   = QColor(30, 30, 35)
KEY_ACTIVE_W   = QColor(100, 200, 255)
KEY_ACTIVE_B   = QColor(60, 140, 220)
KEY_LABEL      = QColor(120, 120, 140)
KEY_WIDTH      = 48
ROLL_MIN_PITCH = 21
ROLL_MAX_PITCH = 108
VISIBLE_SECS   = 8.0


class PianoRollWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.notes = []
        self.duration = 0.0
        self.current_time = 0.0
        self.active_pitches = set()
        self.setMinimumHeight(300)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setStyleSheet("background: transparent;")

    def load_midi(self, midi_path):
        self.notes = []
        self.current_time = 0.0
        self.active_pitches = set()
        try:
            mid = mido.MidiFile(midi_path)
            tempo = 500000
            ticks_per_beat = mid.ticks_per_beat
            active = {}
            abs_tick = 0

            for track in mid.tracks:
                abs_tick = 0
                for msg in track:
                    abs_tick += msg.time
                    t = mido.tick2second(abs_tick, ticks_per_beat, tempo)
                    if msg.type == "set_tempo":
                        tempo = msg.tempo
                    elif msg.type == "note_on" and msg.velocity > 0:
                        active[msg.note] = (t, msg.velocity)
                    elif msg.type == "note_off" or (msg.type == "note_on" and msg.velocity == 0):
                        if msg.note in active:
                            start, vel = active.pop(msg.note)
                            self.notes.append((start, t, msg.note, vel))

            self.duration = max((n[1] for n in self.notes), default=0.0) + 1.0
        except Exception:
            pass
        self.update()

    def set_time(self, t):
        self.current_time = t
        self.active_pitches = {n[2] for n in self.notes if n[0] <= t <= n[1]}
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, False)

        w = self.width()
        h = self.height()
        roll_x = KEY_WIDTH
        roll_w = w - KEY_WIDTH

        num_pitches = ROLL_MAX_PITCH - ROLL_MIN_PITCH + 1
        lane_h = max(4, h / num_pitches)

        t_start = max(0.0, self.current_time - VISIBLE_SECS * 0.25)
        t_end = t_start + VISIBLE_SECS
        px_per_sec = roll_w / VISIBLE_SECS

        def pitch_y(pitch):
            idx = ROLL_MAX_PITCH - pitch
            return idx * lane_h

        def time_x(t):
            return roll_x + (t - t_start) * px_per_sec

        painter.fillRect(0, 0, w, h, ROLL_BG)

        for pitch in range(ROLL_MIN_PITCH, ROLL_MAX_PITCH + 1):
            ni = pitch % 12
            y = pitch_y(pitch)
            if ni in BLACK_KEYS:
                painter.fillRect(int(roll_x), int(y), roll_w, max(1, int(lane_h)), LANE_ALT)

        beat_dur = 0.5
        b = int(t_start / beat_dur)
        while b * beat_dur < t_end:
            bx = time_x(b * beat_dur)
            col = GRID_BEAT if b % 4 == 0 else GRID_LINE
            painter.setPen(QPen(col, 1))
            painter.drawLine(int(bx), 0, int(bx), h)
            b += 1

        painter.setPen(QPen(GRID_LINE, 1))
        for pitch in range(ROLL_MIN_PITCH, ROLL_MAX_PITCH + 1):
            y = pitch_y(pitch)
            painter.drawLine(roll_x, int(y), w, int(y))

        for start, end, pitch, vel in self.notes:
            if end < t_start or start > t_end:
                continue
            x1 = max(roll_x, time_x(start))
            x2 = min(w, time_x(end))
            y = pitch_y(pitch)
            nw = max(2, x2 - x1)
            nh = max(2, lane_h - 1)
            ni = pitch % 12
            is_active = pitch in self.active_pitches
            base = NOTE_HIGHLIGHT if is_active else (BLACK_KEY_NOTE if ni in BLACK_KEYS else NOTE_COLOR)
            painter.fillRect(int(x1), int(y) + 1, int(nw), int(nh) - 1, base)
            painter.setPen(QPen(NOTE_BORDER if is_active else NOTE_COLOR.lighter(130), 1))
            painter.drawRect(int(x1), int(y) + 1, int(nw) - 1, int(nh) - 2)

        playhead_x = time_x(self.current_time)
        painter.setPen(QPen(PLAYHEAD_COLOR, 2))
        painter.drawLine(int(playhead_x), 0, int(playhead_x), h)

        for pitch in range(ROLL_MIN_PITCH, ROLL_MAX_PITCH + 1):
            ni = pitch % 12
            y = pitch_y(pitch)
            is_active = pitch in self.active_pitches
            if ni in BLACK_KEYS:
                col = KEY_ACTIVE_B if is_active else KEY_BLACK_BG
                painter.fillRect(0, int(y) + 1, KEY_WIDTH - 2, max(1, int(lane_h) - 1), col)
            else:
                col = KEY_ACTIVE_W if is_active else KEY_WHITE_BG
                painter.fillRect(0, int(y) + 1, KEY_WIDTH - 2, max(1, int(lane_h) - 1), col)
                if ni == 0:
                    painter.setPen(QPen(KEY_LABEL, 1))
                    painter.setFont(QFont("Courier", max(7, int(lane_h * 0.7))))
                    octave = (pitch // 12) - 1
                    painter.drawText(2, int(y) + int(lane_h), f"C{octave}")

        painter.setPen(QPen(GRID_LINE, 1))
        painter.drawLine(KEY_WIDTH - 1, 0, KEY_WIDTH - 1, h)