import sys
import sounddevice as sd
import soundfile as sf
import subprocess
import tempfile
import os
import numpy as np
from datetime import datetime
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QRadioButton, QCheckBox, QGroupBox,
    QTextEdit, QButtonGroup, QMessageBox, QFileDialog, QSpinBox
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt6.QtGui import QFont, QPalette, QColor

NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
SAMPLE_RATE = 44100
CHANNELS = 1


class RecordingThread(QThread):
    status_update = pyqtSignal(str)
    finished = pyqtSignal(np.ndarray)

    def __init__(self):
        super().__init__()
        self.is_recording = False
        self.audio_data = []

    def run(self):
        self.is_recording = True
        self.audio_data = []

        def callback(indata, frames, time, status):
            if status:
                self.status_update.emit(f"Recording status: {status}")
            if self.is_recording:
                self.audio_data.append(indata.copy())

        try:
            stream = sd.InputStream(
                samplerate=SAMPLE_RATE,
                channels=CHANNELS,
                dtype="float32",
                callback=callback
            )
            stream.start()

            while self.is_recording:
                self.msleep(100)

            stream.stop()
            stream.close()

            if self.audio_data:
                audio = np.concatenate(self.audio_data, axis=0)
                self.finished.emit(audio)
            else:
                self.finished.emit(np.array([]))

        except Exception as e:
            self.status_update.emit(f"Recording error: {str(e)}")
            self.finished.emit(np.array([]))

    def stop(self):
        self.is_recording = False


class ProcessingThread(QThread):
    status_update = pyqtSignal(str)
    result_ready = pyqtSignal(str)
    error_occurred = pyqtSignal(str)
    finished = pyqtSignal()

    def __init__(self, audio, device, save_midi, fixed_velocity=None, fixed_pitch_bend=None, extend=False):
        super().__init__()
        self.audio = audio
        self.device = device
        self.save_midi = save_midi
        self.fixed_velocity = fixed_velocity
        self.fixed_pitch_bend = fixed_pitch_bend
        self.extend = extend

    def run(self):
        if len(self.audio) == 0:
            self.status_update.emit("No audio recorded.")
            self.finished.emit()
            return

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            tmp_wav = f.name

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        midi_path = f"detected_notes_{timestamp}.mid"

        try:
            sf.write(tmp_wav, self.audio, SAMPLE_RATE)
            self.status_update.emit("Running Transkun analysis...")

            success = self.run_transkun(tmp_wav, midi_path, self.device)

            if not success:
                self.status_update.emit("Transcription failed.")
                return

            self.status_update.emit("Transcription complete!")

            if self.fixed_velocity is not None:
                self.apply_fixed_velocity(midi_path, self.fixed_velocity)

            if self.fixed_pitch_bend is not None:
                self.apply_fixed_pitch_bend(midi_path, self.fixed_pitch_bend)

            notes = self.read_midi_notes(midi_path)
            self.display_notes(notes)

            if self.save_midi:
                self.result_ready.emit(f"\nMIDI saved to: {midi_path}")
                self.result_ready.emit("Open it in GarageBand, Logic, or MuseScore.")
            else:
                if os.path.exists(midi_path):
                    os.unlink(midi_path)
                self.result_ready.emit("\nMIDI file removed.")

        except Exception as e:
            self.error_occurred.emit(f"Error: {str(e)}")
        finally:
            if os.path.exists(tmp_wav):
                os.unlink(tmp_wav)
            self.finished.emit()

    def run_transkun(self, wav_path, midi_out_path, device="cpu"):
        try:
            import shutil
            transkun_exe = shutil.which("transkun")
            if transkun_exe is None:
                self.error_occurred.emit("Transkun executable not found. Make sure it is installed.")
                return False

            cmd = [transkun_exe, wav_path, midi_out_path, "--device", device]
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                self.error_occurred.emit(f"Transkun error:\n{result.stderr}")
                return False
            return True
        except Exception as e:
            self.error_occurred.emit(f"Transkun error: {str(e)}")
            return False
        
    def apply_acoustic_sustain(self, midi_path, audio):
        try:
            import mido
        except ImportError:
            return

        mid = mido.MidiFile(midi_path)
        sr = SAMPLE_RATE
        ticks_per_beat = mid.ticks_per_beat

        def get_energy(t, window=0.1):
            start = int(max(0, (t - window) * sr))
            end = int(min(len(audio), (t + window) * sr))
            if start >= end:
                return 0
            return float(np.sqrt(np.mean(audio[start:end].flatten() ** 2)))

        for track in mid.tracks:
            current_tick = 0
            tempo = 500000
            pedal_on = False
            insertions = []

            for msg in track:
                current_tick += msg.time
                seconds = mido.tick2second(current_tick, ticks_per_beat, tempo)

                if msg.type == "set_tempo":
                    tempo = msg.tempo

                elif msg.type == "note_off" or (msg.type == "note_on" and msg.velocity == 0):
                    energy_at = get_energy(seconds)
                    energy_after = get_energy(seconds + 0.05)

                    sustained = energy_at > 0 and energy_after / energy_at > 0.5

                    if sustained and not pedal_on:
                        pedal_on = True
                        pedal_tick = current_tick
                        insertions.append(mido.Message(
                            'control_change', channel=0,
                            control=64, value=127,
                            time=pedal_tick
                        ))

                    elif not sustained and pedal_on:
                        pedal_on = False
                        pedal_off_tick = current_tick
                        insertions.append(mido.Message(
                            'control_change', channel=0,
                            control=64, value=0,
                            time=pedal_off_tick
                        ))

            if pedal_on:
                insertions.append(mido.Message(
                    'control_change', channel=0,
                    control=64, value=0,
                    time=current_tick
                ))

            for msg in insertions:
                track.append(msg)

            track.sort(key=lambda m: m.time)

        mid.save(midi_path)
        self.result_ready.emit("Applied acoustic sustain pedal (CC64).")

    def apply_fixed_velocity(self, midi_path, velocity):
        try:
            import mido
        except ImportError:
            self.error_occurred.emit("Install mido: pip install mido")
            return
        mid = mido.MidiFile(midi_path)
        for track in mid.tracks:
            for msg in track:
                if msg.type == "note_on" and msg.velocity > 0:
                    msg.velocity = velocity
        mid.save(midi_path)
        self.result_ready.emit(f"Applied fixed velocity: {velocity}")

    def apply_fixed_pitch_bend(self, midi_path, pitch_bend):
        try:
            import mido
        except ImportError:
            self.error_occurred.emit("Install mido: pip install mido")
            return
        mid = mido.MidiFile(midi_path)
        for track in mid.tracks:
            for i in reversed(range(len(track))):
                if track[i].type == "pitchwheel":
                    track.pop(i)
            if len(track) > 0:
                track.insert(0, mido.Message('pitchwheel', pitch=pitch_bend, time=0))
        mid.save(midi_path)
        self.result_ready.emit(f"Applied fixed pitch bend: {pitch_bend}")

    def read_midi_notes(self, midi_path):
        try:
            import mido
        except ImportError:
            self.error_occurred.emit("Install mido: pip install mido")
            return []

        mid = mido.MidiFile(midi_path)
        notes = []
        tempo = 500000
        ticks_per_beat = mid.ticks_per_beat
        active = {}

        for track in mid.tracks:
            current_tick = 0
            for msg in track:
                current_tick += msg.time
                seconds = mido.tick2second(current_tick, ticks_per_beat, tempo)
                if msg.type == "set_tempo":
                    tempo = msg.tempo
                elif msg.type == "note_on" and msg.velocity > 0:
                    active[msg.note] = (seconds, msg.velocity)
                elif msg.type == "note_off" or (msg.type == "note_on" and msg.velocity == 0):
                    if msg.note in active:
                        start, vel = active.pop(msg.note)
                        notes.append((start, seconds, msg.note, vel))

        return sorted(notes, key=lambda n: n[0])

    def midi_to_note_name(self, midi_pitch):
        name = NOTE_NAMES[int(midi_pitch) % 12]
        octave = (int(midi_pitch) // 12) - 1
        return f"{name}{octave}"

    def display_notes(self, notes):
        if not notes:
            self.result_ready.emit("No notes detected.")
            return

        self.result_ready.emit(f"{'Start':>7}  {'End':>7}  {'Note':<6}  {'MIDI':>4}  {'Velocity':>8}")
        self.result_ready.emit("-" * 45)

        for start, end, pitch, velocity in notes:
            note_name = self.midi_to_note_name(pitch)
            self.result_ready.emit(f"{start:>6.2f}s  {end:>6.2f}s  {note_name:<6}  {pitch:>4}  {velocity:>8}")

        self.result_ready.emit(f"\nTotal notes detected: {len(notes)}")


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

        # Device selection
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

        # Velocity options
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

        # Pitch bend options
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

        # Sustain pedal + save MIDI options
        self.extend_check = QCheckBox("Extend notes with sustain pedal (recommended)")
        self.extend_check.setFont(QFont("Helvetica", 10))
        self.extend_check.setChecked(True)
        main_layout.addWidget(self.extend_check)

        self.save_midi_check = QCheckBox("Save MIDI file after transcription")
        self.save_midi_check.setFont(QFont("Helvetica", 10))
        self.save_midi_check.setChecked(True)
        main_layout.addWidget(self.save_midi_check)

        # Results
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
                            f"File is {sr}Hz but librosa is not installed for resampling.\n"
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
        self.processing_thread.start()

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

    def update_status(self, message):
        self.status_label.setText(message)

    def append_result(self, text):
        self.results_text.append(text)

    def show_error(self, message):
        self.results_text.append(f"ERROR: {message}")
        QMessageBox.critical(self, "Error", message)


def main():
    app = QApplication(sys.argv)
    window = PianoDetectorUI()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()