import subprocess
import tempfile
import os
import shutil
import numpy as np
import soundfile as sf
from datetime import datetime
from PyQt6.QtCore import QThread, pyqtSignal, QObject

SAMPLE_RATE = 44100
NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']


class MidiWorker(QObject):
    done = pyqtSignal()

    def __init__(self, fn):
        super().__init__()
        self.fn = fn
        

    def run(self):
        self.fn()
        self.done.emit()


class ProcessingThread(QThread):
    status_update = pyqtSignal(str)
    result_ready = pyqtSignal(str)
    error_occurred = pyqtSignal(str)
    finished = pyqtSignal()
    midi_ready = pyqtSignal(str)

    def __init__(self, audio, device, save_midi, fixed_velocity=None, fixed_pitch_bend=None, extend=False, output_folder=None):
        super().__init__()
        self.audio = audio
        self.device = device
        self.save_midi = save_midi
        self.fixed_velocity = fixed_velocity
        self.fixed_pitch_bend = fixed_pitch_bend
        self.extend = extend
        self.output_folder = output_folder or os.path.dirname(os.path.abspath(__file__))

    def run(self):
        if len(self.audio) == 0:
            self.status_update.emit("No audio recorded.")
            self.finished.emit()
            return

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            tmp_wav = f.name
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        midi_path = os.path.join(self.output_folder, f"detected_notes_{timestamp}.mid")

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

            self.apply_acoustic_sustain(midi_path, self.audio)

            notes = self.read_midi_notes(midi_path)
            self.display_notes(notes)

            if self.save_midi:
                self.result_ready.emit(f"\nMIDI saved to: {midi_path}")
                self.result_ready.emit("Open it in GarageBand, Logic, or MuseScore.")
                self.midi_ready.emit(midi_path)
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
        
    # something is wrong here
    def apply_acoustic_sustain(self, midi_path, audio):
        print("acoustic sustain is working!")
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
            # Convert delta times to absolute ticks
            abs_tick = 0
            for msg in track:
                abs_tick += msg.time
                msg.time = abs_tick

            tempo = 500000
            pedal_on = False
            insertions = []

            for msg in track:
                seconds = mido.tick2second(msg.time, ticks_per_beat, tempo)
                if msg.type == "set_tempo":
                    tempo = msg.tempo
                elif msg.type == "note_off" or (msg.type == "note_on" and msg.velocity == 0):
                    energy_at = get_energy(seconds)
                    energy_after = get_energy(seconds + 0.05)
                    sustained = energy_at > 0 and energy_after / energy_at > 0.2

                    if sustained and not pedal_on:
                        pedal_on = True
                        insertions.append(mido.Message(
                            'control_change', channel=0,
                            control=64, value=127,
                            time=msg.time
                        ))
                    elif not sustained and pedal_on:
                        pedal_on = False
                        insertions.append(mido.Message(
                            'control_change', channel=0,
                            control=64, value=0,
                            time=msg.time
                        ))

            if pedal_on and track:
                last_tick = max(msg.time for msg in track)
                insertions.append(mido.Message(
                    'control_change', channel=0,
                    control=64, value=0,
                    time=last_tick
                ))

            for msg in insertions:
                track.append(msg)

            # Sort by absolute tick
            track.sort(key=lambda m: m.time)

            # Convert back to delta times
            prev_tick = 0
            for msg in track:
                abs_t = msg.time
                msg.time = abs_t - prev_tick
                prev_tick = abs_t

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