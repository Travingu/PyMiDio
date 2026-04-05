go to terminal and run python main.py (for windows idk about mac)

Necessary installations
pip install pyqt6 sounddevice soundfile numpy mido pygame torch 

Transkun
pip install git+https://github.com/Yujia-Yan/Transkun.git

I got this, idk if it is necessary
https://aka.ms/vs/17/release/vc_redist.x64.exe

Download ffmpeg if you want to import mp3 files

If you want to make sure that audio files are 44100 Hz (Transkun is trained on 44100 Hz audio)
pip install librosa

To make sure everything works correctly before it runs (DLLs specifically) (speaking from experience)
pip install msvc-runtime

IF YOU ARE IMPORTING TO FL STUDIO
Midi imports do not naturally import CC64 (sustain pedal). It sounds the sustain is not working but it is just FL studio.
read the link below
https://forum.image-line.com/viewtopic.php?t=305901