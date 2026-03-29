import sys
from PyQt6.QtWidgets import QApplication
from piano_detector_ui import PianoDetectorUI
 
 
def main():
    app = QApplication(sys.argv)
    window = PianoDetectorUI()
    window.show()
    sys.exit(app.exec())
 
 
if __name__ == "__main__":
    main()
 