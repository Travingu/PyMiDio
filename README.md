This repository uses Transkun open source model and adds a GUI over the system which allows for recording and other settings
which are not available to just using Transkun (fixed velocity, pitch bend, sustain). It also includes an Audio and Midi Player (Visual Piano Roll!) as well.
This allows users to replay audio from either the recorded audio (.wav only) or a selected Midi File. The Midi Player imports the notes onto a visual piano roll which helps users visualize what is being played.

https://github.com/yujia-yan/transkun

Instructions
go to terminal and run python main.py (for windows idk about mac)

Necessary installations
pip install pyqt6 sounddevice soundfile numpy mido pygame torch 

Transkun
pip install git+https://github.com/Yujia-Yan/Transkun.git

To make sure everything works correctly before it runs (DLLs specifically) (speaking from experience)
pip install msvc-runtime

IF YOU ARE IMPORTING TO FL STUDIO
Midi imports do not naturally import CC64 (sustain pedal). It sounds the sustain is not working but it is just FL studio.
read the link below
https://forum.image-line.com/viewtopic.php?t=305901
