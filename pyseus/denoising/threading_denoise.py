from PySide2.QtCore import QThread, Signal
import numpy
from pyseus.denoising.tv import TV


# One could remove TV Class here and pass to init function also the denoise_gen itself 
# if this class shouldnt import TV and be retained as generic

class ThreadingDenoised(QThread):
    output = Signal(numpy.ndarray)
    def __init__(self, parent_thr, function, dataset, params):
        QThread.__init__(self, parent=parent_thr)
        self.data_noisy = dataset
        self.data_denoised = None
        self.function = function
        self.params = params
        
    
    
    # cannot pass arguments to run, but to the contructor of the class 
    def run(self):
        # *args: Passing a Function Using with an arbitrary number of positional argument
        tv_class = TV()
        self.data_denoised = tv_class.tv_denoising_gen(self.function, self.data_noisy, self.params)
        self.output.emit(self.data_denoised)

        

