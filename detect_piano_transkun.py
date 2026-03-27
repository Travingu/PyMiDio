import sounddevice as sd
import soundfile as sf
import subprocess
import tempfile
import os
import sys
 
NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
 
SAMPLE_RATE = 44100
CHANNELS = 1
 
 
def midi_to_note_name(midi_pitch):
    name = NOTE_NAMES[int(midi_pitch) % 12]
    octave = (int(midi_pitch) // 12) - 1
    return f"{name}{octave}"
 
 
def record_audio(duration: float):
    print(f"\nRecording for {duration} seconds... Play now!")
    audio = sd.rec(
        int(duration * SAMPLE_RATE),
        samplerate=SAMPLE_RATE,
        channels=CHANNELS,
        dtype="float32"
    )
    sd.wait()
    print("Recording done.\n")
    return audio
 
 
def run_transkun(self, wav_path, midi_out_path, device="cpu"):
    try:
        from transkun.transcribe import transcribeFile
        transcribeFile(wav_path, midi_out_path, device=device)
        return True
    except Exception as e:
        self.error_occurred.emit(f"Transkun error: {str(e)}")
        return False
 
 
def read_midi_notes(midi_path: str):
    try:
        import mido
    except ImportError:
        print("\nInstall mido to print note events: pip install mido")
        return []
 
    mid = mido.MidiFile(midi_path)
    notes = []
    tempo = 500000
    ticks_per_beat = mid.ticks_per_beat
    active = {}
    current_tick = 0
 
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
 
 
def print_note_events(notes):
    if not notes:
        print("No notes detected.")
        return
 
    print(f"{'Start':>7}  {'End':>7}  {'Note':<6}  {'MIDI':>4}  {'Velocity':>8}")
    print("-" * 45)
 
    for start, end, pitch, velocity in notes:
        note_name = midi_to_note_name(pitch)
        print(f"{start:>6.2f}s  {end:>6.2f}s  {note_name:<6}  {pitch:>4}  {velocity:>8}")
 
    print(f"\nTotal notes detected: {len(notes)}")
 
 
def main():
    print("=== Piano Note Detector (Transkun) ===")
 
    try:
        duration_str = input("Recording duration in seconds [default 5]: ").strip()
        duration = float(duration_str) if duration_str else 5.0
    except ValueError:
        duration = 5.0
 
    keep_str = input("Keep the MIDI file after printing? (y/n) [default y]: ").strip().lower()
    keep_midi = keep_str != "n"
 
    device_str = input("Device — cpu or cuda [default cpu]: ").strip().lower()
    device = device_str if device_str in ("cpu", "cuda") else "cpu"
 
    audio = record_audio(duration)
 
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        tmp_wav = f.name
 
    midi_path = "detected_notes_transkun.mid"
 
    try:
        sf.write(tmp_wav, audio, SAMPLE_RATE)
        success = run_transkun(tmp_wav, midi_path, device)
    finally:
        os.unlink(tmp_wav)
 
    if not success:
        return
 
    print(f"\nMIDI saved to: {midi_path}")
    notes = read_midi_notes(midi_path)
    print_note_events(notes)
 
    if not keep_midi:
        os.unlink(midi_path)
        print("MIDI file removed.")
    else:
        print(f"\nOpen {midi_path} in GarageBand, Logic, or MuseScore.")
 
 
if __name__ == "__main__":
    main()