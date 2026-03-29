import sys
import subprocess
import tempfile
import os
import numpy as np
import soundfile as sf
import sounddevice as sd
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QRadioButton, QCheckBox, QGroupBox,
    QTextEdit, QButtonGroup, QMessageBox, QFileDialog, QSpinBox,
    QSlider, QSplitter, QFrame, QScrollArea
)
from PyQt6.QtCore import Qt, QThread, QTimer
from PyQt6.QtGui import QFont, QPalette, QColor, QScreen

from recording_thread import RecordingThread, SAMPLE_RATE, CHANNELS
from processing_thread import ProcessingThread, MidiWorker
from piano_roll_widget import PianoRollWidget

NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']


def btn_style(bg="#2a82da", hover="#3a92ea", pressed="#1a72ca"):
    return f"""
        QPushButton {{
            background-color: {bg}; color: white;
            border: none; border-radius: 6px;
            padding: 8px;
        }}
        QPushButton:hover {{ background-color: {hover}; }}
        QPushButton:pressed {{ background-color: {pressed}; }}
        QPushButton:disabled {{ background-color: #444444; color: #777777; }}
    """


class PianoDetectorUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Piano Note Detector — Transkun")

        self.is_recording = False
        self.recording_thread = None
        self.processing_thread = None
        self.recording_seconds = 0
        self.current_audio = None
        self.is_playing = False
        self.playback_stream = None
        self.playback_position = 0.0
        self.current_midi_path = None
        self._audio_position = 0
        self._audio_total = 0
        self._audio_data = None

        self.setup_dark_theme()
        self.setup_ui()

        self.timer = QTimer()
        self.timer.timeout.connect(self.update_timer)

        self.playback_timer = QTimer()
        self.playback_timer.timeout.connect(self.update_playback_ui)

        self.showMaximized()

    def setup_dark_theme(self):
        palette = QPalette()
        palette.setColor(QPalette.ColorRole.Window, QColor(22, 22, 30))
        palette.setColor(QPalette.ColorRole.WindowText, QColor(220, 220, 230))
        palette.setColor(QPalette.ColorRole.Base, QColor(30, 30, 40))
        palette.setColor(QPalette.ColorRole.AlternateBase, QColor(35, 35, 45))
        palette.setColor(QPalette.ColorRole.ToolTipBase, QColor(20, 20, 28))
        palette.setColor(QPalette.ColorRole.ToolTipText, QColor(220, 220, 230))
        palette.setColor(QPalette.ColorRole.Text, QColor(220, 220, 230))
        palette.setColor(QPalette.ColorRole.Button, QColor(35, 35, 48))
        palette.setColor(QPalette.ColorRole.ButtonText, QColor(220, 220, 230))
        palette.setColor(QPalette.ColorRole.Highlight, QColor(42, 130, 218))
        palette.setColor(QPalette.ColorRole.HighlightedText, QColor(255, 255, 255))
        QApplication.setPalette(palette)

    def setup_ui(self):
        root = QWidget()
        self.setCentralWidget(root)
        root_layout = QVBoxLayout(root)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(0)

        # Title bar
        title_bar = QWidget()
        title_bar.setFixedHeight(52)
        title_bar.setStyleSheet("background: #12121a; border-bottom: 1px solid #2a2a3a;")
        title_layout = QHBoxLayout(title_bar)
        title_layout.setContentsMargins(24, 0, 24, 0)
        title_lbl = QLabel("Piano Note Detector")
        title_lbl.setFont(QFont("Helvetica", 16, QFont.Weight.Bold))
        title_lbl.setStyleSheet("color: #ffffff;")
        title_layout.addWidget(title_lbl)
        title_layout.addStretch()
        root_layout.addWidget(title_bar)

        # Main splitter
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setStyleSheet("QSplitter::handle { background: #2a2a3a; width: 2px; }")
        root_layout.addWidget(splitter)

        # ── LEFT PANEL ──────────────────────────────────────────
        left_scroll = QScrollArea()
        left_scroll.setWidgetResizable(True)
        left_scroll.setFrameShape(QFrame.Shape.NoFrame)
        left_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        left_scroll.setStyleSheet("background: #16161e;")

        left = QWidget()
        left.setStyleSheet("background: #16161e;")
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(24, 24, 24, 24)
        left_layout.setSpacing(14)
        left_scroll.setWidget(left)

        # Section label
        def section_label(text):
            l = QLabel(text)
            l.setFont(QFont("Helvetica", 11, QFont.Weight.Bold))
            l.setStyleSheet("color: #6a6a8a; letter-spacing: 1px; margin-top: 8px;")
            return l

        left_layout.addWidget(section_label("RECORDING"))

        self.record_button = QPushButton("Start Recording")
        self.record_button.setFont(QFont("Helvetica", 13))
        self.record_button.setMinimumHeight(50)
        self.record_button.setStyleSheet(btn_style())
        self.record_button.clicked.connect(self.toggle_recording)
        left_layout.addWidget(self.record_button)

        self.load_button = QPushButton("Load Audio File (.wav / .mp3)")
        self.load_button.setFont(QFont("Helvetica", 12))
        self.load_button.setMinimumHeight(44)
        self.load_button.setStyleSheet(btn_style("#2a2a3a", "#3a3a4a", "#1a1a2a"))
        self.load_button.clicked.connect(self.load_audio_file)
        left_layout.addWidget(self.load_button)

        self.status_label = QLabel("Ready to record or load audio")
        self.status_label.setFont(QFont("Helvetica", 10))
        self.status_label.setStyleSheet("color: #8888aa;")
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        left_layout.addWidget(self.status_label)

        self.time_label = QLabel("00:00")
        self.time_label.setFont(QFont("Helvetica", 36, QFont.Weight.Bold))
        self.time_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.time_label.setStyleSheet("color: #2a82da;")
        left_layout.addWidget(self.time_label)

        left_layout.addWidget(section_label("ANALYSIS"))

        self.analyze_button = QPushButton("Analyze")
        self.analyze_button.setFont(QFont("Helvetica", 13))
        self.analyze_button.setMinimumHeight(50)
        self.analyze_button.setEnabled(False)
        self.analyze_button.setStyleSheet(btn_style("#1a6a1a", "#2a8a2a", "#0a5a0a"))
        self.analyze_button.clicked.connect(self.analyze_audio)
        left_layout.addWidget(self.analyze_button)

        # Device
        device_group = QGroupBox("Device")
        device_group.setFont(QFont("Helvetica", 10))
        device_group.setStyleSheet("QGroupBox { color: #aaaacc; border: 1px solid #2a2a3a; border-radius: 6px; margin-top: 6px; padding: 8px; } QGroupBox::title { subcontrol-origin: margin; left: 8px; }")
        device_layout = QHBoxLayout()
        self.device_group = QButtonGroup()
        self.cpu_radio = QRadioButton("CPU")
        self.cpu_radio.setFont(QFont("Helvetica", 10))
        self.cpu_radio.setChecked(True)
        self.cpu_radio.setStyleSheet("""
            QRadioButton {
                color: #aaaacc;
                spacing: 8px;
            }
            QRadioButton::indicator {
                width: 18px;
                height: 18px;
                border-radius: 9px;
                border: 2px solid #6a6a8a;
                background: transparent;
            }
            QRadioButton::indicator:checked {
                background-color: #2a82da;
                border: 2px solid #ffffff;
            }
            QRadioButton::indicator:unchecked {
                background: transparent;
                border: 2px solid #6a6a8a;
            }
        """)
        self.cuda_radio = QRadioButton("CUDA (GPU)")
        self.cuda_radio.setFont(QFont("Helvetica", 10))
        self.cuda_radio.setStyleSheet("""
            QRadioButton {
                color: #aaaacc;
                spacing: 8px;
            }
            QRadioButton::indicator {
                width: 18px;
                height: 18px;
                border-radius: 9px;
                border: 2px solid #6a6a8a;
                background: transparent;
            }
            QRadioButton::indicator:checked {
                background-color: #2a82da;
                border: 2px solid #ffffff;
            }
            QRadioButton::indicator:unchecked {
                background: transparent;
                border: 2px solid #6a6a8a;
            }
        """)
        self.device_group.addButton(self.cpu_radio)
        self.device_group.addButton(self.cuda_radio)
        device_layout.addWidget(self.cpu_radio)
        device_layout.addWidget(self.cuda_radio)
        device_group.setLayout(device_layout)
        left_layout.addWidget(device_group)

        # Velocity
        velocity_group = QGroupBox("Velocity")
        velocity_group.setFont(QFont("Helvetica", 10))
        velocity_group.setStyleSheet(device_group.styleSheet())
        velocity_layout = QVBoxLayout()
        self.original_velocity_radio = QRadioButton("Original")
        self.original_velocity_radio.setFont(QFont("Helvetica", 10))
        self.original_velocity_radio.setChecked(True)
        self.original_velocity_radio.setStyleSheet("""
            QRadioButton {
                color: #aaaacc;
                spacing: 8px;
            }
            QRadioButton::indicator {
                width: 18px;
                height: 18px;
                border-radius: 9px;
                border: 2px solid #6a6a8a;
                background: transparent;
            }
            QRadioButton::indicator:checked {
                background-color: #2a82da;
                border: 2px solid #ffffff;
            }
            QRadioButton::indicator:unchecked {
                background: transparent;
                border: 2px solid #6a6a8a;
            }
        """)
        self.fixed_velocity_radio = QRadioButton("Fixed:")
        self.fixed_velocity_radio.setFont(QFont("Helvetica", 10))
        self.fixed_velocity_radio.setStyleSheet("""
            QRadioButton {
                color: #aaaacc;
                spacing: 8px;
            }
            QRadioButton::indicator {
                width: 18px;
                height: 18px;
                border-radius: 9px;
                border: 2px solid #6a6a8a;
                background: transparent;
            }
            QRadioButton::indicator:checked {
                background-color: #2a82da;
                border: 2px solid #ffffff;
            }
            QRadioButton::indicator:unchecked {
                background: transparent;
                border: 2px solid #6a6a8a;
            }
        """)
        self.velocity_spinbox = QSpinBox()
        self.velocity_spinbox.setFont(QFont("Helvetica", 10))
        self.velocity_spinbox.setRange(1, 127)
        self.velocity_spinbox.setValue(100)
        self.velocity_spinbox.setEnabled(False)
        self.fixed_velocity_radio.toggled.connect(self.velocity_spinbox.setEnabled)
        velocity_layout.addWidget(self.original_velocity_radio)
        fv = QHBoxLayout()
        fv.addWidget(self.fixed_velocity_radio)
        fv.addWidget(self.velocity_spinbox)
        fv.addStretch()
        velocity_layout.addLayout(fv)
        velocity_group.setLayout(velocity_layout)
        left_layout.addWidget(velocity_group)

        # Pitch bend
        pitch_group = QGroupBox("Pitch Bend")
        pitch_group.setFont(QFont("Helvetica", 10))
        pitch_group.setStyleSheet(device_group.styleSheet())
        pitch_layout = QVBoxLayout()
        self.original_pitch_bend_radio = QRadioButton("Original")
        self.original_pitch_bend_radio.setFont(QFont("Helvetica", 10))
        self.original_pitch_bend_radio.setChecked(True)
        self.original_pitch_bend_radio.setStyleSheet("""
            QRadioButton {
                color: #aaaacc;
                spacing: 8px;
            }
            QRadioButton::indicator {
                width: 18px;
                height: 18px;
                border-radius: 9px;
                border: 2px solid #6a6a8a;
                background: transparent;
            }
            QRadioButton::indicator:checked {
                background-color: #2a82da;
                border: 2px solid #ffffff;
            }
            QRadioButton::indicator:unchecked {
                background: transparent;
                border: 2px solid #6a6a8a;
            }
        """)
        self.fixed_pitch_bend_radio = QRadioButton("Fixed:")
        self.fixed_pitch_bend_radio.setFont(QFont("Helvetica", 10))
        self.fixed_pitch_bend_radio.setStyleSheet("""
            QRadioButton {
                color: #aaaacc;
                spacing: 8px;
            }
            QRadioButton::indicator {
                width: 18px;
                height: 18px;
                border-radius: 9px;
                border: 2px solid #6a6a8a;
                background: transparent;
            }
            QRadioButton::indicator:checked {
                background-color: #2a82da;
                border: 2px solid #ffffff;
            }
            QRadioButton::indicator:unchecked {
                background: transparent;
                border: 2px solid #6a6a8a;
            }
        """)
        self.pitch_bend_spinbox = QSpinBox()
        self.pitch_bend_spinbox.setFont(QFont("Helvetica", 10))
        self.pitch_bend_spinbox.setRange(-8192, 8191)
        self.pitch_bend_spinbox.setValue(0)
        self.pitch_bend_spinbox.setEnabled(False)
        self.fixed_pitch_bend_radio.toggled.connect(self.pitch_bend_spinbox.setEnabled)
        pitch_layout.addWidget(self.original_pitch_bend_radio)
        fp = QHBoxLayout()
        fp.addWidget(self.fixed_pitch_bend_radio)
        fp.addWidget(self.pitch_bend_spinbox)
        fp.addStretch()
        pitch_layout.addLayout(fp)
        pitch_group.setLayout(pitch_layout)
        left_layout.addWidget(pitch_group)

        self.extend_check = QCheckBox("Sustain pedal detection")
        self.extend_check.setFont(QFont("Helvetica", 10))
        self.extend_check.setChecked(True)
        self.extend_check.setStyleSheet("""
            QCheckBox {
                color: #aaaacc;
                spacing: 8px;
            }
            QCheckBox::indicator {
                width: 18px;
                height: 18px;
                border-radius: 4px;
                border: 2px solid #6a6a8a;
                background: transparent;
            }
            QCheckBox::indicator:checked {
                background-color: #2a82da;
                border: 2px solid #ffffff;
            }
            QCheckBox::indicator:unchecked {
                background: transparent;
                border: 2px solid #6a6a8a;
            }
        """)
        left_layout.addWidget(self.extend_check)

        self.save_midi_check = QCheckBox("Save MIDI file")
        self.save_midi_check.setFont(QFont("Helvetica", 10))
        self.save_midi_check.setChecked(True)
        self.save_midi_check.setStyleSheet("""
            QCheckBox {
                color: #aaaacc;
                spacing: 8px;
            }
            QCheckBox::indicator {
                width: 18px;
                height: 18px;
                border-radius: 4px;
                border: 2px solid #6a6a8a;
                background: transparent;
            }
            QCheckBox::indicator:checked {
                background-color: #2a82da;
                border: 2px solid #ffffff;
            }
            QCheckBox::indicator:unchecked {
                background: transparent;
                border: 2px solid #6a6a8a;
            }
        """)
        left_layout.addWidget(self.save_midi_check)

        left_layout.addWidget(section_label("RESULTS"))

        self.results_text = QTextEdit()
        self.results_text.setReadOnly(True)
        self.results_text.setFont(QFont("Courier", 9))
        self.results_text.setMinimumHeight(200)
        self.results_text.setStyleSheet("""
            QTextEdit {
                background: #0e0e16; color: #aaaacc;
                border: 1px solid #2a2a3a; border-radius: 6px;
                padding: 8px;
            }
        """)
        left_layout.addWidget(self.results_text)
        left_layout.addStretch()

        splitter.addWidget(left_scroll)

        # ── RIGHT PANEL ─────────────────────────────────────────
        right = QWidget()
        right.setStyleSheet("background: #12121a;")
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(24, 24, 24, 24)
        right_layout.setSpacing(14)

        right_layout.addWidget(section_label("PLAYBACK"))

        # Mode toggle
        mode_layout = QHBoxLayout()
        self.playback_mode_group = QButtonGroup()
        self.audio_mode_radio = QRadioButton("Audio")
        self.audio_mode_radio.setFont(QFont("Helvetica", 10))
        self.audio_mode_radio.setChecked(True)
        self.audio_mode_radio.setStyleSheet("""
            QRadioButton {
                color: #aaaacc;
                spacing: 8px;
            }
            QRadioButton::indicator {
                width: 18px;
                height: 18px;
                border-radius: 9px;
                border: 2px solid #6a6a8a;
                background: transparent;
            }
            QRadioButton::indicator:checked {
                background-color: #2a82da;
                border: 2px solid #ffffff;
            }
            QRadioButton::indicator:unchecked {
                background: transparent;
                border: 2px solid #6a6a8a;
            }
        """)
        self.midi_mode_radio = QRadioButton("MIDI")
        self.midi_mode_radio.setFont(QFont("Helvetica", 10))
        self.midi_mode_radio.setStyleSheet("""
            QRadioButton {
                color: #aaaacc;
                spacing: 8px;
            }
            QRadioButton::indicator {
                width: 18px;
                height: 18px;
                border-radius: 9px;
                border: 2px solid #6a6a8a;
                background: transparent;
            }
            QRadioButton::indicator:checked {
                background-color: #2a82da;
                border: 2px solid #ffffff;
            }
            QRadioButton::indicator:unchecked {
                background: transparent;
                border: 2px solid #6a6a8a;
            }
        """)
        self.playback_mode_group.addButton(self.audio_mode_radio)
        self.playback_mode_group.addButton(self.midi_mode_radio)
        mode_layout.addWidget(self.audio_mode_radio)
        mode_layout.addWidget(self.midi_mode_radio)
        mode_layout.addStretch()
        right_layout.addLayout(mode_layout)

        # MIDI browse row
        midi_row = QHBoxLayout()
        self.midi_path_label = QLabel("No MIDI file selected")
        self.midi_path_label.setFont(QFont("Helvetica", 10))
        self.midi_path_label.setStyleSheet("color: #666688;")
        self.browse_midi_button = QPushButton("Browse")
        self.browse_midi_button.setFont(QFont("Helvetica", 10))
        self.browse_midi_button.setFixedWidth(80)
        self.browse_midi_button.setStyleSheet(btn_style("#2a2a3a", "#3a3a4a", "#1a1a2a"))
        self.browse_midi_button.clicked.connect(self.browse_midi_file)
        midi_row.addWidget(self.midi_path_label, 1)
        midi_row.addWidget(self.browse_midi_button)
        right_layout.addLayout(midi_row)

        # Progress + time
        self.playback_slider = QSlider(Qt.Orientation.Horizontal)
        self.playback_slider.setRange(0, 1000)
        self.playback_slider.setValue(0)
        self.playback_slider.sliderMoved.connect(self.on_slider_moved)
        self.playback_slider.setStyleSheet("""
            QSlider::groove:horizontal { height: 4px; background: #2a2a3a; border-radius: 2px; }
            QSlider::handle:horizontal { background: #2a82da; width: 14px; height: 14px; margin: -5px 0; border-radius: 7px; }
            QSlider::sub-page:horizontal { background: #2a82da; border-radius: 2px; }
        """)
        right_layout.addWidget(self.playback_slider)

        self.playback_time_label = QLabel("0:00 / 0:00")
        self.playback_time_label.setFont(QFont("Helvetica", 10))
        self.playback_time_label.setStyleSheet("color: #6666aa;")
        self.playback_time_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        right_layout.addWidget(self.playback_time_label)

        # Play/stop
        self.play_button = QPushButton("Play")
        self.play_button.setFont(QFont("Helvetica", 13))
        self.play_button.setMinimumHeight(50)
        self.play_button.setStyleSheet(btn_style())
        self.play_button.clicked.connect(self.toggle_playback)
        right_layout.addWidget(self.play_button)

        self.playback_status_label = QLabel("No audio or MIDI loaded")
        self.playback_status_label.setFont(QFont("Helvetica", 10))
        self.playback_status_label.setStyleSheet("color: #6666aa;")
        self.playback_status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        right_layout.addWidget(self.playback_status_label)

        # Divider
        divider = QFrame()
        divider.setFrameShape(QFrame.Shape.HLine)
        divider.setStyleSheet("color: #2a2a3a;")
        right_layout.addWidget(divider)

        # Piano roll label
        roll_lbl = QLabel("PIANO ROLL")
        roll_lbl.setFont(QFont("Helvetica", 11, QFont.Weight.Bold))
        roll_lbl.setStyleSheet("color: #6a6a8a; letter-spacing: 1px;")
        right_layout.addWidget(roll_lbl)

        # Piano roll
        self.piano_roll = PianoRollWidget()
        self.piano_roll.setMinimumHeight(400)
        right_layout.addWidget(self.piano_roll, 1)

        splitter.addWidget(right)

        # Set splitter proportions — left ~35%, right ~65%
        splitter.setSizes([350, 650])

    # ── RECORDING ────────────────────────────────────────────────

    def toggle_recording(self):
        if not self.is_recording:
            self.start_recording()
        else:
            self.stop_recording()

    def start_recording(self):
        self.is_recording = True
        self.recording_seconds = 0
        self.current_audio = None
        self.record_button.setText("Stop Recording")
        self.record_button.setStyleSheet(btn_style("#aa2222", "#cc3333", "#881111"))
        self.status_label.setText("Recording... Play now!")
        self.time_label.setText("00:00")
        self.results_text.clear()
        self.analyze_button.setEnabled(False)

        self.recording_thread = RecordingThread()
        self.recording_thread.status_update.connect(self.update_status)
        self.recording_thread.finished.connect(self.on_recording_finished)
        self.recording_thread.start()
        self.timer.start(1000)

    def stop_recording(self):
        self.is_recording = False
        self.timer.stop()
        if self.recording_thread:
            self.recording_thread.stop()
        self.record_button.setText("Start Recording")
        self.record_button.setStyleSheet(btn_style())
        self.status_label.setText("Recording stopped. Click Analyze.")

    def update_timer(self):
        if self.is_recording:
            self.recording_seconds += 1
            m = self.recording_seconds // 60
            s = self.recording_seconds % 60
            self.time_label.setText(f"{m:02d}:{s:02d}")

    def on_recording_finished(self, audio):
        if len(audio) == 0:
            self.update_status("No audio recorded.")
            self.record_button.setEnabled(True)
            return
        self.current_audio = audio
        self.status_label.setText("Recording complete. Click Analyze.")
        self.analyze_button.setEnabled(True)
        self.record_button.setEnabled(True)

    # ── FILE LOADING ─────────────────────────────────────────────

    def load_audio_file(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Load Audio File", "",
            "Audio Files (*.wav *.mp3);;WAV Files (*.wav);;MP3 Files (*.mp3);;All Files (*)"
        )
        if not file_path:
            return

        self.results_text.clear()
        self.status_label.setText(f"Loading: {os.path.basename(file_path)}")
        self.analyze_button.setEnabled(False)

        try:
            if file_path.lower().endswith('.mp3'):
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                    tmp_wav = f.name
                cmd = ['ffmpeg', '-i', file_path, '-ar', str(SAMPLE_RATE), '-ac', str(CHANNELS), '-y', tmp_wav]
                result = subprocess.run(cmd, capture_output=True, text=True)
                if result.returncode != 0:
                    QMessageBox.critical(self, "Error", f"Failed to convert MP3:\n{result.stderr}")
                    if os.path.exists(tmp_wav):
                        os.unlink(tmp_wav)
                    return
                audio, _ = sf.read(tmp_wav, dtype='float32')
                os.unlink(tmp_wav)
            else:
                audio, sr = sf.read(file_path, dtype='float32')
                if sr != SAMPLE_RATE:
                    try:
                        import librosa
                        if len(audio.shape) > 1:
                            audio = audio.mean(axis=1)
                        audio = librosa.resample(audio, orig_sr=sr, target_sr=SAMPLE_RATE)
                    except ImportError:
                        QMessageBox.warning(self, "Warning",
                            "librosa not installed — resampling skipped.\npip install librosa")

            if len(audio.shape) > 1:
                audio = audio.mean(axis=1)

            self.current_audio = audio
            duration = len(audio) / SAMPLE_RATE
            self.status_label.setText(f"Loaded ({duration:.1f}s). Click Analyze.")
            self.analyze_button.setEnabled(True)

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load:\n{str(e)}")

    # ── ANALYSIS ─────────────────────────────────────────────────

    def analyze_audio(self):
        if self.current_audio is None:
            QMessageBox.warning(self, "Warning", "No audio loaded.")
            return

        self.analyze_button.setEnabled(False)
        self.load_button.setEnabled(False)
        self.record_button.setEnabled(False)
        self.status_label.setText("Analyzing...")

        device = "cpu" if self.cpu_radio.isChecked() else "cuda"
        save_midi = self.save_midi_check.isChecked()
        extend = self.extend_check.isChecked()
        fixed_velocity = self.velocity_spinbox.value() if self.fixed_velocity_radio.isChecked() else None
        fixed_pitch_bend = self.pitch_bend_spinbox.value() if self.fixed_pitch_bend_radio.isChecked() else None

        self.processing_thread = ProcessingThread(
            self.current_audio, device, save_midi,
            fixed_velocity, fixed_pitch_bend, extend
        )
        self.processing_thread.status_update.connect(self.update_status)
        self.processing_thread.result_ready.connect(self.append_result)
        self.processing_thread.error_occurred.connect(self.show_error)
        self.processing_thread.finished.connect(self.on_processing_finished)
        self.processing_thread.midi_ready.connect(self.on_midi_ready)
        self.processing_thread.start()

    def on_processing_finished(self):
        self.record_button.setEnabled(True)
        self.load_button.setEnabled(True)
        self.analyze_button.setEnabled(True)
        self.status_label.setText("Done. Ready for next recording.")

    def on_midi_ready(self, midi_path):
        self.current_midi_path = midi_path
        self.midi_path_label.setText(os.path.basename(midi_path))
        self.midi_path_label.setStyleSheet("color: #aaaacc;")
        self.playback_status_label.setText("MIDI ready — press Play.")
        self.piano_roll.load_midi(midi_path)

    # ── PLAYBACK ─────────────────────────────────────────────────

    def browse_midi_file(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select MIDI file", "",
            "MIDI Files (*.mid *.midi);;All Files (*)"
        )
        if path:
            self.current_midi_path = path
            self.midi_path_label.setText(os.path.basename(path))
            self.midi_path_label.setStyleSheet("color: #aaaacc;")
            self.playback_status_label.setText("MIDI loaded — press Play.")
            self.piano_roll.load_midi(path)

    def toggle_playback(self):
        if self.is_playing:
            self.stop_playback()
        else:
            self.start_playback()

    def start_playback(self):
        if self.audio_mode_radio.isChecked():
            if self.current_audio is None:
                QMessageBox.warning(self, "Warning", "No audio loaded.")
                return
            self.play_audio()
        else:
            if self.current_midi_path is None:
                QMessageBox.warning(self, "Warning", "No MIDI file selected.")
                return
            self.play_midi()

    def play_audio(self):
        audio = self.current_audio
        if len(audio.shape) == 1:
            audio = audio.reshape(-1, 1)

        total_samples = len(self.current_audio)
        start_sample = int(self.playback_position * total_samples)
        self._audio_position = start_sample
        self._audio_data = audio
        self._audio_total = total_samples

        self.is_playing = True
        self.play_button.setText("Stop")
        self.playback_status_label.setText("Playing audio...")

        def callback(outdata, frames, time, status):
            end = self._audio_position + frames
            chunk = self._audio_data[self._audio_position:end]
            if len(chunk) == 0:
                outdata[:] = 0
                raise sd.CallbackStop()
            if len(chunk) < frames:
                outdata[:len(chunk)] = chunk
                outdata[len(chunk):] = 0
                raise sd.CallbackStop()
            else:
                outdata[:] = chunk
            self._audio_position = end

        def on_finished():
            self.is_playing = False
            self.play_button.setText("Play")
            self.playback_timer.stop()
            self.playback_position = 0
            self.playback_slider.setValue(0)
            self.playback_status_label.setText("Playback finished.")

        self.playback_stream = sd.OutputStream(
            samplerate=SAMPLE_RATE,
            channels=self._audio_data.shape[1],
            dtype='float32',
            callback=callback,
            finished_callback=on_finished
        )
        self.playback_stream.start()
        self.playback_timer.start(100)

    def play_midi(self):
        try:
            import pygame
            import pygame.midi
        except ImportError:
            QMessageBox.critical(self, "Error", "Install pygame: pip install pygame")
            return

        pygame.init()
        pygame.midi.init()
        self.is_playing = True
        self.play_button.setText("Stop")
        self.playback_status_label.setText("Playing MIDI...")

        try:
            player = pygame.midi.Output(pygame.midi.get_default_output_id())
            player.set_instrument(0)

            import mido
            mid = mido.MidiFile(self.current_midi_path)
            self._midi_player = player

            def run_midi():
                for msg in mid.play():
                    if not self.is_playing:
                        break
                    if msg.type == 'note_on':
                        player.note_on(msg.note, msg.velocity)
                    elif msg.type == 'note_off':
                        player.note_off(msg.note, msg.velocity)
                    elif msg.type == 'control_change':
                        player.write_short(0xB0, msg.control, msg.value)
                player.close()
                pygame.midi.quit()
                self.is_playing = False
                self.play_button.setText("Play")
                self.playback_status_label.setText("MIDI playback finished.")

            self._midi_thread = QThread()
            self._midi_worker = MidiWorker(run_midi)
            self._midi_worker.moveToThread(self._midi_thread)
            self._midi_thread.started.connect(self._midi_worker.run)
            self._midi_worker.done.connect(self._midi_thread.quit)
            self._midi_thread.start()

            self._midi_start_time = None
            self._midi_duration = sum(
                msg.time for track in mid.tracks for msg in track
                if hasattr(msg, 'time')
            )
            self.playback_timer.start(100)

        except Exception as e:
            self.is_playing = False
            self.play_button.setText("Play")
            QMessageBox.critical(self, "Error", f"MIDI playback error: {str(e)}")

    def stop_playback(self):
        self.is_playing = False
        self.play_button.setText("Play")
        self.playback_timer.stop()
        if self.playback_stream is not None:
            self.playback_stream.stop()
            self.playback_stream.close()
            self.playback_stream = None
        self.playback_status_label.setText("Playback stopped.")

    def update_playback_ui(self):
        if self._audio_data is None or not self.is_playing:
            return
        total = self._audio_total
        pos = self._audio_position
        if total > 0:
            pct = pos / total
            self.playback_slider.setValue(int(pct * 1000))
            elapsed = pos / SAMPLE_RATE
            total_secs = total / SAMPLE_RATE
            self.playback_time_label.setText(
                f"{int(elapsed//60)}:{int(elapsed%60):02d} / "
                f"{int(total_secs//60)}:{int(total_secs%60):02d}"
            )
            self.piano_roll.set_time(elapsed)

    def on_slider_moved(self, value):
        self.playback_position = value / 1000.0
        if self.is_playing:
            self.stop_playback()

    # ── HELPERS ──────────────────────────────────────────────────

    def update_status(self, message):
        self.status_label.setText(message)

    def append_result(self, text):
        self.results_text.append(text)

    def show_error(self, message):
        self.results_text.append(f"ERROR: {message}")
        QMessageBox.critical(self, "Error", message)