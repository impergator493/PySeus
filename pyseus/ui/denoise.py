from PySide2.QtWidgets import QButtonGroup, QDialog, QDialogButtonBox, QFormLayout, QLineEdit, QMainWindow, QAction, QLabel, QFileDialog, \
                              QFrame, QPushButton, QRadioButton, QVBoxLayout, QHBoxLayout, QWidget

from .view import ViewWidget

from pyseus.denoising.tv import TV
import scipy.io
import matplotlib.pyplot as plt

class DialogDenoise(QDialog):

    # parent Window has to be added?
    def __init__(self,app):
        super().__init__()
     
        self.app = app
        self.window_denoised = DenoisedWindow(self.app)

        self.alpha = 0
        self.lambd = 0
        self.iter = 0
              

        self.dialog = QDialog()
        vlayout = QVBoxLayout()
        hlayout = QHBoxLayout()
        self.dialog.setWindowTitle("Denoise")
        
        # subgroup of radio buttons for data selection
        self.lab_data_sel = QLabel("Data Selection")
        self.grp_data_sel = QButtonGroup()
        self.btn_curr_slice = QRadioButton("Current Slice")
        self.btn_curr_slice.setChecked(True)
        self.btn_all_slices = QRadioButton("Whole Dataset")
        self.grp_data_sel.addButton(self.btn_curr_slice,1)
        self.grp_data_sel.addButton(self.btn_all_slices,2)
        self.btn_all_slices_2D = QRadioButton("2D")
        self.btn_all_slices_3D = QRadioButton("3D")

        # subgroup of radio buttons to dataset selection
        self.lab_denoise_type = QLabel("Denoising Type")
        self.grp_tv_type = QButtonGroup()
        self.btn_tv_L1 = QRadioButton("L1")
        self.btn_tv_L2 = QRadioButton("L2")
        self.btn_tv_ROF = QRadioButton("HuberROF")
        self.grp_tv_type.addButton(self.btn_tv_L1, 1)
        self.grp_tv_type.addButton(self.btn_tv_L2, 2)
        self.grp_tv_type.addButton(self.btn_tv_ROF, 3)

    
        # form layout for parameter input for denoising algorithm
        form = QFormLayout()
        self.qline_lambd = QLineEdit()
        #TV_lambda.setStyleSheet("color: white; background-color: darkgray")
        form.addRow("Lambda",self.qline_lambd)
        self.qline_iter = QLineEdit()
        #TV_iter.setStyleSheet("color: white; background-color: darkgray")
        form.addRow("Iterations",self.qline_iter)
        self.qline_alpha = QLineEdit()
        #TV_alpha.setStyleSheet("color: white; background-color: darkgray")
        form.addRow("Alpha",self.qline_alpha)

        
        # function without brackets just connects the function, but does not call it
        self.box_btns = QDialogButtonBox()
        self.box_btns.setStandardButtons(QDialogButtonBox.Cancel | QDialogButtonBox.Ok)
        self.box_btns.accepted.connect(self.signal_ok)
        self.box_btns.rejected.connect(lambda:self.dialog.close())
   

        # organize items on GUI
        vlayout.addWidget(self.lab_data_sel)
        vlayout.addWidget(self.btn_curr_slice)
        vlayout.addWidget(self.btn_all_slices)
        vlayout.addWidget(self.lab_denoise_type)
        vlayout.addWidget(self.btn_tv_L1)
        vlayout.addWidget(self.btn_tv_L2)
        vlayout.addWidget(self.btn_tv_ROF)
        vlayout.addWidget(self.box_btns)

        hlayout.addLayout(vlayout)
        hlayout.addLayout(form)
        
        self.dialog.setLayout(hlayout)
        #dialog.setStyleSheet('color: white')
        self.dialog.setStyleSheet("QLineEdit"
                                    "{"
                                    "color: white; background : darkgray;"
                                    "}" 
                                "QLabel"
                                    "{"
                                    "color: white;"
                                    "}"
                                "QRadioButton"
                                    "{"
                                    "color: white;"
                                    "}"
                                )
        self.dialog.show()

        


    def signal_ok(self):
        self.alpha = float(self.qline_alpha.text())
        self.lambd = float(self.qline_lambd.text())
        self.iter = int(self.qline_iter.text())  

        


        #print(self.grp_tv_type.checkedId())
        
        noisy = scipy.io.loadmat('./tests/cameraman_noise.mat')['im']
        denoise = TV()
        
        denoised = denoise.tv_denoising_L2(noisy,self.lambd,self.iter)

        plt.figure(figsize=(16,10))
        plt.subplot(121)
        plt.imshow(noisy, cmap=plt.cm.gray)
        plt.axis('off')
        plt.title('noisy', fontsize=20)
        plt.subplot(122)
        plt.imshow(denoised, cmap=plt.cm.gray)
        plt.axis('off')
        plt.title('denoised', fontsize=20) 

        plt.get_current_fig_manager().window.showMaximized()
        plt.show()

        self.window_denoised.show()


class DenoisedWindow(QWidget):

    def __init__(self,app):
        super().__init__()
        
        self.view = ViewWidget(app)
        wrapper = QFrame(self)
        wrapper.setLayout(QHBoxLayout())
        wrapper.layout().addWidget(self.view)
        #self.setCentralWidget(wrapper)
