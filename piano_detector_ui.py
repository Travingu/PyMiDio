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
    QTabWidget, QSlider
)
from PyQt6.QtCore import Qt, QThread, QTimer
from PyQt6.QtGui import QFont, QPalette, QColor

from recording_thread import RecordingThread

SAMPLE_RATE = 44100
CHANNELS = 1
from processing_thread import ProcessingThread, MidiWorker

NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']


class PianoDetectorUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Piano Note Detector (Transkun)")
        self.setFixedSize(550, 800)

        self.is_recording = False
        self.recording_thread = None
        self.processing_thread = None
        self.recording_seconds = 0
        self.current_audio = None

        self.setup_dark_theme()
        self.create_widgets()

        self.timer = QTimer()
        self.timer.timeout.connect(self.update_timer)

    def setup_dark_theme(self):
        palette = QPalette()
        palette.setColor(QPalette.ColorRole.Window, QColor(30, 30, 30))
        palette.setColor(QPalette.ColorRole.WindowText, QColor(255, 255, 255))
        palette.setColor(QPalette.ColorRole.Base, QColor(45, 45, 45))
        palette.setColor(QPalette.ColorRole.AlternateBase, QColor(50, 50, 50))
        palette.setColor(QPalette.ColorRole.ToolTipBase, QColor(25, 25, 25))
        palette.setColor(QPalette.ColorRole.ToolTipText, QColor(255, 255, 255))
        palette.setColor(QPalette.ColorRole.Text, QColor(255, 255, 255))
        palette.setColor(QPalette.ColorRole.Button, QColor(50, 50, 50))
        palette.setColor(QPalette.ColorRole.ButtonText, QColor(255, 255, 255))
        palette.setColor(QPalette.ColorRole.BrightText, QColor(255, 0, 0))
        palette.setColor(QPalette.ColorRole.Link, QColor(42, 130, 218))
        palette.setColor(QPalette.ColorRole.Highlight, QColor(42, 130, 218))
        palette.setColor(QPalette.ColorRole.HighlightedText, QColor(0, 0, 0))
        QApplication.setPalette(palette)

    def create_widgets(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)

        self.tabs = QTabWidget()
        main_layout.addWidget(self.tabs)

        self.main_tab = QWidget()
        self.tabs.addTab(self.main_tab, "Record & Analyze")
        self.create_main_tab()

        self.playback_tab = QWidget()
        self.tabs.addTab(self.playback_tab, "Playback")
        self.create_playback_tab()

    def create_main_tab(self):
        main_layout = QVBoxLayout(self.main_tab)
        main_layout.setSpacing(12)
        main_layout.setContentsMargins(20, 20, 20, 20)

        title_label = QLabel("Piano Note Detector")
        title_label.setFont(QFont("Helvetica", 20, QFont.Weight.Bold))
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        main_layout.addWidget(title_label)

        btn_style = """
            QPushButton {
                background-color: #2a82da; color: white;
                border: none; border-radius: 5px;
            }
            QPushButton:hover { background-color: #3a92ea; }
            QPushButton:pressed { background-color: #1a72ca; }
            QPushButton:disabled { background-color: #555555; color: #888888; }
        """

        self.record_button = QPushButton("Start Recording")
        self.record_button.setFont(QFont("Helvetica", 12))
        self.record_button.setMinimumHeight(45)
        self.record_button.setStyleSheet(btn_style)
        self.record_button.clicked.connect(self.toggle_recording)
        main_layout.addWidget(self.record_button)

        self.load_button = QPushButton("Load Audio File (.wav / .mp3)")
        self.load_button.setFont(QFont("Helvetica", 12))
        self.load_button.setMinimumHeight(45)
        self.load_button.setStyleSheet(btn_style)
        self.load_button.clicked.connect(self.load_audio_file)
        main_layout.addWidget(self.load_button)

        self.analyze_button = QPushButton("Analyze")
        self.analyze_button.setFont(QFont("Helvetica", 12))
        self.analyze_button.setMinimumHeight(45)
        self.analyze_button.setEnabled(False)
        self.analyze_button.setStyleSheet(btn_style)
        self.analyze_button.clicked.connect(self.analyze_audio)
        main_layout.addWidget(self.analyze_button)

        self.status_label = QLabel("Ready to record or load audio file")
        self.status_label.setFont(QFont("Helvetica", 10))
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        main_layout.addWidget(self.status_label)

        self.time_label = QLabel("00:00")
        self.time_label.setFont(QFont("Helvetica", 28, QFont.Weight.Bold))
        self.time_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.time_label.setStyleSheet("color: #2a82da;")
        main_layout.addWidget(self.time_label)

        device_group = QGroupBox("Device Selection")
        device_group.setFont(QFont("Helvetica", 10))
        device_layout = QHBoxLayout()
        self.device_group = QButtonGroup()
        self.cpu_radio = QRadioButton("CPU")
        self.cpu_radio.setFont(QFont("Helvetica", 10))
        self.cpu_radio.setChecked(True)
        self.cuda_radio = QRadioButton("CUDA (GPU)")
        self.cuda_radio.setFont(QFont("Helvetica", 10))
        self.device_group.addButton(self.cpu_radio)
        self.device_group.addButton(self.cuda_radio)
        device_layout.addWidget(self.cpu_radio)
        device_layout.addWidget(self.cuda_radio)
        device_group.setLayout(device_layout)
        main_layout.addWidget(device_group)

        velocity_group = QGroupBox("Velocity Options")
        velocity_group.setFont(QFont("Helvetica", 10))
        velocity_layout = QVBoxLayout()
        self.original_velocity_radio = QRadioButton("Original velocity")
        self.original_velocity_radio.setFont(QFont("Helvetica", 10))
        self.original_velocity_radio.setChecked(True)
        self.fixed_velocity_radio = QRadioButton("Fixed velocity:")
        self.fixed_velocity_radio.setFont(QFont("Helvetica", 10))
        self.velocity_spinbox = QSpinBox()
        self.velocity_spinbox.setFont(QFont("Helvetica", 10))
        self.velocity_spinbox.setRange(1, 127)
        self.velocity_spinbox.setValue(100)
        self.velocity_spinbox.setEnabled(False)
        self.fixed_velocity_radio.toggled.connect(self.velocity_spinbox.setEnabled)
        velocity_layout.addWidget(self.original_velocity_radio)
        fixed_vel_layout = QHBoxLayout()
        fixed_vel_layout.addWidget(self.fixed_velocity_radio)
        fixed_vel_layout.addWidget(self.velocity_spinbox)
        fixed_vel_layout.addStretch()
        velocity_layout.addLayout(fixed_vel_layout)
        velocity_group.setLayout(velocity_layout)
        main_layout.addWidget(velocity_group)

        pitch_bend_group = QGroupBox("Pitch Bend Options")
        pitch_bend_group.setFont(QFont("Helvetica", 10))
        pitch_bend_layout = QVBoxLayout()
        self.original_pitch_bend_radio = QRadioButton("Original pitch bend")
        self.original_pitch_bend_radio.setFont(QFont("Helvetica", 10))
        self.original_pitch_bend_radio.setChecked(True)
        self.fixed_pitch_bend_radio = QRadioButton("Fixed pitch bend:")
        self.fixed_pitch_bend_radio.setFont(QFont("Helvetica", 10))
        self.pitch_bend_spinbox = QSpinBox()
        self.pitch_bend_spinbox.setFont(QFont("Helvetica", 10))
        self.pitch_bend_spinbox.setRange(-8192, 8191)
        self.pitch_bend_spinbox.setValue(0)
        self.pitch_bend_spinbox.setEnabled(False)
        self.fixed_pitch_bend_radio.toggled.connect(self.pitch_bend_spinbox.setEnabled)
        pitch_bend_layout.addWidget(self.original_pitch_bend_radio)
        fixed_bend_layout = QHBoxLayout()
        fixed_bend_layout.addWidget(self.fixed_pitch_bend_radio)
        fixed_bend_layout.addWidget(self.pitch_bend_spinbox)
        fixed_bend_layout.addStretch()
        pitch_bend_layout.addLayout(fixed_bend_layout)
        pitch_bend_group.setLayout(pitch_bend_layout)
        main_layout.addWidget(pitch_bend_group)

        self.extend_check = QCheckBox("Extend notes with sustain pedal (recommended)")
        self.extend_check.setFont(QFont("Helvetica", 10))
        self.extend_check.setChecked(True)
        main_layout.addWidget(self.extend_check)

        self.save_midi_check = QCheckBox("Save MIDI file after transcription")
        self.save_midi_check.setFont(QFont("Helvetica", 10))
        self.save_midi_check.setChecked(True)
        main_layout.addWidget(self.save_midi_check)

        results_group = QGroupBox("Results")
        results_group.setFont(QFont("Helvetica", 10))
        results_layout = QVBoxLayout()
        self.results_text = QTextEdit()
        self.results_text.setReadOnly(True)
        self.results_text.setFont(QFont("Courier", 9))
        self.results_text.setStyleSheet("""
            QTextEdit {
                background-color: #2d2d2d; color: #ffffff;
                border: 1px solid #3d3d3d; border-radius: 3px;
            }
        """)
        results_layout.addWidget(self.results_text)
        results_group.setLayout(results_layout)
        main_layout.addWidget(results_group)

    def create_playback_tab(self):
        layout = QVBoxLayout(self.playback_tab)
        layout.setSpacing(12)
        layout.setContentsMargins(20, 20, 20, 20)

        self.playback_mode_group = QButtonGroup()
        mode_layout = QHBoxLayout()
        self.audio_mode_radio = QRadioButton("Play audio")
        self.audio_mode_radio.setFont(QFont("Helvetica", 10))
        self.audio_mode_radio.setChecked(True)
        self.midi_mode_radio = QRadioButton("Play MIDI")
        self.midi_mode_radio.setFont(QFont("Helvetica", 10))
        self.playback_mode_group.addButton(self.audio_mode_radio)
        self.playback_mode_group.addButton(self.midi_mode_radio)
        mode_layout.addWidget(self.audio_mode_radio)
        mode_layout.addWidget(self.midi_mode_radio)
        layout.addLayout(mode_layout)

        midi_row = QHBoxLayout()
        self.midi_path_label = QLabel("No MIDI file selected")
        self.midi_path_label.setFont(QFont("Helvetica", 10))
        self.midi_path_label.setStyleSheet("color: #888888;")
        self.browse_midi_button = QPushButton("Browse")
        self.browse_midi_button.setFont(QFont("Helvetica", 10))
        self.browse_midi_button.clicked.connect(self.browse_midi_file)
        midi_row.addWidget(self.midi_path_label, 1)
        midi_row.addWidget(self.browse_midi_button)
        layout.addLayout(midi_row)

        self.playback_slider = QSlider(Qt.Orientation.Horizontal)
        self.playback_slider.setRange(0, 1000)
        self.playback_slider.setValue(0)
        self.playback_slider.sliderMoved.connect(self.on_slider_moved)
        layout.addWidget(self.playback_slider)

        self.playback_time_label = QLabel("0:00 / 0:00")
        self.playback_time_label.setFont(QFont("Helvetica", 10))
        self.playback_time_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.playback_time_label)

        btn_style = """
            QPushButton {
                background-color: #2a82da; color: white;
                border: none; border-radius: 5px;
            }
            QPushButton:hover { background-color: #3a92ea; }
            QPushButton:pressed { background-color: #1a72ca; }
            QPushButton:disabled { background-color: #555555; color: #888888; }
        """
        self.play_button = QPushButton("Play")
        self.play_button.setFont(QFont("Helvetica", 12))
        self.play_button.setMinimumHeight(45)
        self.play_button.setStyleSheet(btn_style)
        self.play_button.clicked.connect(self.toggle_playback)
        layout.addWidget(self.play_button)

        self.playback_status_label = QLabel("No audio or MIDI loaded")
        self.playback_status_label.setFont(QFont("Helvetica", 10))
        self.playback_status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.playback_status_label)

        layout.addStretch()

        self.is_playing = False
        self.playback_stream = None
        self.playback_position = 0
        self.current_midi_path = None
        self._audio_position = 0
        self._audio_total = 0
        self._audio_data = None

        self.playback_timer = QTimer()
        self.playback_timer.timeout.connect(self.update_playback_ui)

    def browse_midi_file(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select MIDI file", "",
            "MIDI Files (*.mid *.midi);;All Files (*)"
        )
        if path:
            self.current_midi_path = path
            self.midi_path_label.setText(os.path.basename(path))
            self.midi_path_label.setStyleSheet("color: #ffffff;")
            self.playback_status_label.setText("MIDI file loaded. Press Play.")

    def toggle_playback(self):
        if self.is_playing:
            self.stop_playback()
        else:
            self.start_playback()

    def start_playback(self):
        if self.audio_mode_radio.isChecked():
            if self.current_audio is None:
                QMessageBox.warning(self, "Warning", "No audio recorded or loaded yet.")
                return
            self.play_audio()
        else:
            if self.current_midi_path is None:
                QMessageBox.warning(self, "Warning", "No MIDI file selected. Use Browse to pick one.")
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
        self.playback_timer.start(200)

    def play_midi(self):
        try:
            import pygame
            import pygame.midi
        except ImportError:
            QMessageBox.critical(self, "Error", "pygame is required for MIDI playback.\nInstall it with: pip install pygame")
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
            self._midi_mid = mid

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
                f"{int(elapsed//60)}:{int(elapsed%60):02d} / {int(total_secs//60)}:{int(total_secs%60):02d}"
            )

    def on_slider_moved(self, value):
        self.playback_position = value / 1000.0
        if self.is_playing:
            self.stop_playback()

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
        self.record_button.setStyleSheet("""
            QPushButton {
                background-color: #da2a2a; color: white;
                border: none; border-radius: 5px;
            }
            QPushButton:hover { background-color: #ea3a3a; }
            QPushButton:pressed { background-color: #ca1a1a; }
        """)
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
        self.record_button.setStyleSheet("""
            QPushButton {
                background-color: #2a82da; color: white;
                border: none; border-radius: 5px;
            }
            QPushButton:hover { background-color: #3a92ea; }
            QPushButton:pressed { background-color: #1a72ca; }
        """)
        self.status_label.setText("Recording stopped. Click Analyze to process.")

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
                self.status_label.setText("Converting MP3 to WAV...")
                cmd = ['ffmpeg', '-i', file_path, '-ar', str(SAMPLE_RATE), '-ac', str(CHANNELS), '-y', tmp_wav]
                result = subprocess.run(cmd, capture_output=True, text=True)
                if result.returncode != 0:
                    self.status_label.setText("Error converting MP3. Make sure ffmpeg is installed.")
                    QMessageBox.critical(self, "Error", f"Failed to convert MP3:\n{result.stderr}")
                    if os.path.exists(tmp_wav):
                        os.unlink(tmp_wav)
                    return
                audio, _ = sf.read(tmp_wav, dtype='float32')
                os.unlink(tmp_wav)
            else:
                audio, sr = sf.read(file_path, dtype='float32')
                if sr != SAMPLE_RATE:
                    self.status_label.setText(f"Resampling from {sr}Hz to {SAMPLE_RATE}Hz...")
                    try:
                        import librosa
                        if len(audio.shape) > 1:
                            audio = audio.mean(axis=1)
                        audio = librosa.resample(audio, orig_sr=sr, target_sr=SAMPLE_RATE)
                    except ImportError:
                        QMessageBox.warning(self, "Warning",
                            f"File is {sr}Hz but librosa is not installed.\n"
                            "Install it with: pip install librosa\n"
                            "Proceeding anyway — accuracy may be reduced.")

            if len(audio.shape) > 1:
                audio = audio.mean(axis=1)

            self.current_audio = audio
            duration = len(audio) / SAMPLE_RATE
            self.status_label.setText(f"Loaded {os.path.basename(file_path)} ({duration:.1f}s). Click Analyze.")
            self.analyze_button.setEnabled(True)

        except Exception as e:
            self.status_label.setText(f"Error loading file: {str(e)}")
            QMessageBox.critical(self, "Error", f"Failed to load audio file:\n{str(e)}")

    def analyze_audio(self):
        if self.current_audio is None:
            QMessageBox.warning(self, "Warning", "No audio loaded. Please record or load a file first.")
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

    def on_midi_ready(self, midi_path):
        self.current_midi_path = midi_path
        self.midi_path_label.setText(os.path.basename(midi_path))
        self.midi_path_label.setStyleSheet("color: #ffffff;")
        self.playback_status_label.setText("MIDI ready. Switch to Playback tab.")

    def update_timer(self):
        if self.is_recording:
            self.recording_seconds += 1
            minutes = self.recording_seconds // 60
            seconds = self.recording_seconds % 60
            self.time_label.setText(f"{minutes:02d}:{seconds:02d}")

    def on_recording_finished(self, audio):
        if len(audio) == 0:
            self.update_status("No audio recorded.")
            self.record_button.setEnabled(True)
            return
        self.current_audio = audio
        self.status_label.setText("Recording complete. Click Analyze to process.")
        self.analyze_button.setEnabled(True)
        self.record_button.setEnabled(True)

    def on_processing_finished(self):
        self.record_button.setEnabled(True)
        self.load_button.setEnabled(True)
        self.analyze_button.setEnabled(True)
        self.status_label.setText("Ready to record or load audio file")
        self.tabs.setCurrentIndex(1)

    def update_status(self, message):
        self.status_label.setText(message)

    def append_result(self, text):
        self.results_text.append(text)

    def show_error(self, message):
        self.results_text.append(f"ERROR: {message}")
        QMessageBox.critical(self, "Error", message)