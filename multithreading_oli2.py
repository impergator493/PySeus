import time, sys
from PyQt5.QtCore  import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *

class SimulRunner(QObject):
    'Object managing the simulation'

    stepIncreased = pyqtSignal(int, name = 'stepIncreased')
    def __init__(self):
        super(SimulRunner, self).__init__()
        self._step = 0
        self._maxSteps = 200

    def longRunning(self):
        print("Running in Thread: ", QThread.currentThread())
        while self._step  < self._maxSteps:
            if(QThread.currentThread().isInterruptionRequested()):
                print("Interrupt request: ", QThread.currentThread().isInterruptionRequested())
                return
            self._step += 1
            self.stepIncreased.emit(self._step)
            time.sleep(0.1)

class SimulationUi(QDialog):
    'PyQt interface'

    def __init__(self):
        super(SimulationUi, self).__init__()

        self.goButton = QPushButton('Go')
        self.stopButton = QPushButton('Stop')
        self.currentStep = QSpinBox()

        self.layout = QHBoxLayout()
        self.layout.addWidget(self.goButton)
        self.layout.addWidget(self.stopButton)
        self.layout.addWidget(self.currentStep)
        self.setLayout(self.layout)

        self.thread = None

        self.goButton.clicked.connect(self.startJob)
        self.stopButton.clicked.connect(self.stopJob)

    def startJob(self):
        print("Current value of thread: ", self.thread)
        if self.thread is None:
            self.thread = QThread()

        self.simulRunner = SimulRunner()

        self.simulRunner.moveToThread(self.thread)
        self.thread.started.connect(self.simulRunner.longRunning)
        self.simulRunner.stepIncreased.connect(self.currentStep.setValue)
        self.thread.start()

    def stopJob(self):
        print("Current value of thread: ", self.thread)
        if self.thread is not None:
            self.thread.requestInterruption()
            self.thread.quit()
            self.thread.wait()
            print("Thread stopped: ", not self.thread.isRunning())
            del self.thread
            del self.simulRunner
            self.thread = None


if __name__ == '__main__':
    app = QApplication(sys.argv)
    simul = SimulationUi()
    simul.show()
    sys.exit(app.exec_())
