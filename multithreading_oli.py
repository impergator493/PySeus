from PySide2 import QtWidgets
import PySide2.QtCore as Qt
from PySide2.QtWidgets import QButtonGroup, QDialog, QDialogButtonBox, QFormLayout, QLineEdit, QMainWindow, QAction, QLabel, QFileDialog, \
                              QFrame, QPushButton, QRadioButton, QVBoxLayout, QHBoxLayout, QWidget
from PySide2.QtGui import QIcon
from PySide2.QtWidgets import QApplication, QMessageBox


class WorkThread(Qt.QThread):

    threadSignal = Qt.Signal(int) 

    def __init__(self):
        super().__init__()

    def run(self, *args, **kwargs):
        c = 0
        while True:
            Qt.QThread.msleep(100)
            c += 1
            self.threadSignal.emit(c)

class MsgBox(QDialog):
    def __init__(self):
        super().__init__()

        layout     = QVBoxLayout(self)
        self.label = QLabel("")
        layout.addWidget(self.label)
        close_btn  = QPushButton("Close")
        layout.addWidget(close_btn)

        close_btn.clicked.connect(self.close)

        self.setGeometry(900, 65, 400, 80)
        self.setWindowTitle('MsgBox from WorkThread')


class GUI(QWidget):     #(QMainWindow):

    def __init__(self):
        super().__init__()

        layout   = QVBoxLayout(self)
        self.btn = QPushButton("Start thread.")
        layout.addWidget(self.btn)
        self.btn.clicked.connect(self.startExecuting)

        self.msg    = MsgBox()
        self.thread = None
        # Lots of irrelevant code here ...

    # Called when "Start/Stop Executing" button is pressed
    def startExecuting(self, user_script):

        if self.thread is None:
            self.thread = WorkThread()

            self.thread.threadSignal.connect(self.on_threadSignal)
            self.thread.start()

            self.btn.setText("Stop thread")
        else:
            self.thread.terminate()
            self.thread = None
            self.btn.setText("Start thread")


    def on_threadSignal(self, value):
        self.msg.label.setText(str(value))
        if not self.msg.isVisible():
            self.msg.show()


if __name__ == '__main__':
    app = QApplication([])
    mw  = GUI()
    mw.show()
    app.exec_()
