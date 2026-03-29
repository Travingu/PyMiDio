import numpy as np
import sounddevice as sd
from PyQt6.QtCore import QThread, pyqtSignal

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