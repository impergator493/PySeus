from PySide2.QtCore import QThread, Signal
import numpy
from pyseus.denoising.tv import TV
from pyseus.denoising.tgv import TGV
from pyseus.denoising.tgv_3D import TGV_3D


# One could remove TV Class here and pass to init function also the denoise_gen itself 
# if this class shouldnt import TV and be retained as generic

class ThreadingDenoised(QThread):
    output = Signal(numpy.ndarray)
    def __init__(self, parent_thr, tv_class, tv_function, dataset_type, dataset, params):
        QThread.__init__(self, parent=parent_thr)
        self.data_noisy = dataset
        self.data_denoised = None
        self.dataset_type = dataset_type
        self.tv_class = tv_class
        self.tv_function = tv_function
        self.params = params
        
    
    
    # cannot pass arguments to run, but to the constructor of the class 
    def run(self):
        # *args: Passing a Function Using with an arbitrary number of positional argument
        
        if isinstance(self.tv_class,TV):
            self.data_denoised = self.tv_class.tv_denoising_gen(self.tv_function, self.dataset_type, self.data_noisy, self.params)
        elif isinstance(self.tv_class,TGV):
            self.data_denoised = self.tv_class.tgv2_denoising(self.data_noisy, *self.params)
        elif isinstance(self.tv_class,TGV_3D):
            self.data_denoised = self.tv_class.tgv2_3D_denoising(self.data_noisy, *self.params)

        else:
            raise TypeError("No valid denoising class selected")            
        self.output.emit(self.data_denoised)

        

