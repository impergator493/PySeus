from PySide2.QtCore import QThread, Signal
import numpy
from pyseus.processing.tv import TV
from pyseus.processing.tgv import TGV
from pyseus.processing.tgv_reconstruction import TGV_Reco


# One could remove TV Class here and pass to init function also the denoise_gen itself 
# if this class shouldnt import TV and be retained as generic

class ProcessThread(QThread):
    output = Signal(numpy.ndarray)
    def __init__(self, parent_thr, tv_class, tv_function, dataset_type, dataset, params, sp_mask = None, coil_data = None):
        QThread.__init__(self, parent=parent_thr)
        self.dataset = dataset
        self.data_processed = None
        self.dataset_type = dataset_type
        self.tv_class = tv_class
        self.tv_function = tv_function
        self.params = params
        self.sp_mask = sp_mask
        self.coil_data = coil_data
        
    
    
    # cannot pass arguments to run, but to the constructor of the class 
    def run(self):
        # *args: Passing a Function Using with an arbitrary number of positional argument
        
        # denoising
        if isinstance(self.tv_class,TV):
            self.data_processed = self.tv_class.tv_denoising_gen(self.tv_function, self.dataset_type, self.dataset, self.params)
        elif isinstance(self.tv_class,TGV):
            self.data_processed = self.tv_class.tgv2_denoising_gen(self.dataset_type, self.dataset, self.params)
        # reconstruction
        elif isinstance(self.tv_class, TGV_Reco):
            self.data_processed = self.tv_class.tgv2_reconstruction(self.dataset_type,self.dataset, self.coil_data, self.sp_mask, self.params)
        else:
            raise TypeError("No valid denoising class selected")            
        self.output.emit(self.data_processed)

        

