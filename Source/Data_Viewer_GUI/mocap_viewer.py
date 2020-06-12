from PyQt5.QtWidgets import QApplication
import pyqt_viewer

def main():
    qapp = QApplication([])
    main_window = pyqt_viewer.MainWindow()

    # main_window.setWindowTitle("Motion Capture Viewer")
    main_window.show()
    qapp.exec_()

if __name__ == '__main__':
    main()
