from PySide2.QtWidgets import QButtonGroup, QDesktopWidget, QDialog, QDialogButtonBox, QFormLayout, QLayout, QLineEdit, QMainWindow, QAction, QLabel, QFileDialog, \
                              QFrame, QPushButton, QRadioButton, QScrollArea, QVBoxLayout, QHBoxLayout, QWidget

from .view import ViewWidget

from pyseus.denoising.tv import TV
from pyseus.modes.grayscale import Grayscale
from PySide2.QtCore import Qt
import numpy


import scipy.io
import matplotlib.pyplot as plt

class DialogDenoise(QDialog):

    # parent Window has to be added?
    def __init__(self,app):
        super().__init__()
    
        self.app = app
        self.window_denoised = DenoisedWindow(app)



        vlayout = QVBoxLayout()
        hlayout = QHBoxLayout()
        self.setWindowTitle("Denoise")
        
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
        self.box_btns.rejected.connect(lambda:self.close())
   

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
        
        self.setLayout(hlayout)
        #dialog.setStyleSheet('color: white')
        self.setStyleSheet("QLineEdit"
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

        


    def signal_ok(self):
        alpha = float(self.qline_alpha.text())
        lambd = float(self.qline_lambd.text())
        iterations = int(self.qline_iter.text())  

        self.window_denoised.open_dialog(alpha,lambd,iterations)

        # @TODO which other algorithms can be called here, depending on the selected radio button?




class DenoisedWindow(QDialog):

    def __init__(self,app):
        super().__init__()

        self.app = app
        self.denoised = None
        self.slice_id = None 

        self.view = DenoisedViewWidget(self.app, self)
        #self.view.widgetResizable()
        self.box_btns_ok = QDialogButtonBox()
        self.box_btns_ok.setStandardButtons(QDialogButtonBox.Cancel | QDialogButtonBox.Ok)
        self.box_btns_ok.accepted.connect(self.signal_ok)
        self.box_btns_ok.rejected.connect(lambda:self.close())

        self.setLayout(QVBoxLayout())
        self.layout().addWidget(self.view)
        self.layout().addWidget(self.box_btns_ok)


        self.mode = Grayscale()

    def open_dialog(self,alpha,lambd,iterations):

         #print(self.grp_tv_type.checkedId())
        
        #noisy = scipy.io.loadmat('./tests/cameraman_noise.mat')['im']
        denoise = TV()
        
        # get current slice which is displayed, if 3D is selected set -1 slice
        # normally this should be chosen by radiobuttons
        
        #self.slice_id = self.app.get_slice_id()
        self.slice_id = -1
        noisy = self.app.dataset.get_pixeldata(self.slice_id)



        if self.slice_id == -1:
            self.denoised = numpy.zeros(noisy.shape)
            for i in range(0, self.app.dataset.slice_count()):
                self.denoised[i,:,:] = denoise.tv_denoising_L2(noisy[i,:,:],lambd,iterations)
            self.denoised_displayed = self.denoised[(self.app.dataset.slice_count() // 2),:,:]
            self.slice_id = (self.app.dataset.slice_count() // 2)


        else:
            self.denoised = denoise.tv_denoising_L2(noisy,lambd,iterations)
            self.denoised_displayed = self.denoised

        
        self.display_image(self.denoised_displayed)
        # ------------------ Matplot implementation
        # plt.figure(figsize=(16,10))
        # plt.subplot(121)
        # plt.imshow(noisy, cmap=plt.cm.gray)
        # plt.axis('off')
        # plt.title('noisy', fontsize=20)
        # plt.subplot(122)
        # plt.imshow(denoised, cmap=plt.cm.gray)
        # plt.axis('off')
        # plt.title('denoised', fontsize=20) 

        # plt.get_current_fig_manager().window.showMaximized()
        # plt.show()
        # ------------------ Matplot End

        # @TODO
        # shortcut, thats not a good solution
        # should the original grayscale object be used with temporary window
        # or a new grayscale objecte be generated which is independent?
     

    def display_image(self, image):
        
        self.mode.temporary_window(image)
        pixmap = self.mode.get_pixmap(image)
        self.view.set(pixmap)
        #self.setGeometry(600,300,600,600)
        self.show()
        screen_size = QDesktopWidget().screenGeometry()
        self.resize(screen_size.width()*0.3, screen_size.height()*0.3)
        self.view.zoom_fit()

    def resizeEvent(self, event):  # pylint: disable=C0103
        """Keep the viewport centered and adjust zoom on window resize."""
        x_factor = event.size().width() / event.oldSize().width()
        # y_factor = event.size().height() / event.oldSize().height()
        # @TODO x_factor if xf < yf or xf * width * zoom_factor < viewport_x
        self.view.zoom(x_factor, True)

    
    def signal_ok(self):
        
        self.app.set_denoised_dataset(self.denoised)


    def refresh(self, slice_inc):
        
        new_slice = self.slice_id + slice_inc
        if 0 <= new_slice < self.app.dataset.slice_count():
            self.display_image(self.denoised[new_slice])
            self.slice_id = new_slice
        elif new_slice < 0:
            self.slice_id = 0
        else:
            self.slice_id >= self.app.dataset.slice_count()
        
        

class DenoisedViewWidget(QScrollArea):
    """Widget providing an interactive viewport."""

    # @TODO app is obsolete here, generate new class
    def __init__(self, app, dialog):
        QScrollArea.__init__(self)
        self.app = app
        self.dialog = dialog

        self.image = QLabel()
        self.image.setScaledContents(True)
        self.image.setMouseTracking(True)
        

        self.zoom_factor = 1
        """The current zoom factor of the image."""

        self.mouse_action = 0
        """The current action on mouse move.
        Can be *ROI*, *WINDOW* or *PAN*."""

        self.last_position = None
        """The last position, from which mouse events were processed."""

        self.setMouseTracking(True)
        self.setAlignment(Qt.AlignHCenter | Qt.AlignVCenter)
        self.setWidget(self.image)

        # Hide scrollbars
        self.horizontalScrollBar().setStyleSheet("QScrollBar { height: 0 }")
        self.verticalScrollBar().setStyleSheet("QScrollBar { width: 0 }")

    def set(self, pixmap):
        """Display the image in *pixmap*."""
        self.image.setPixmap(pixmap)

    def zoom(self, factor, relative=True):
        """Set the zoom level for the displayed image.

        By default, the new zoom factor will be relative to the current
        zoom factor. If *relative* is set to False, *factor* will be used as
        the new zoom factor."""

        if self.image is None \
                or (relative and (0.1 >= self.zoom_factor * factor >= 100)):
            return

        self.zoom_factor = self.zoom_factor * factor if relative else factor
        self.image.resize(self.zoom_factor * self.image.pixmap().size())

        v_scroll = int(factor * self.verticalScrollBar().value() +
                       ((factor-1) * self.verticalScrollBar().pageStep()/2))
        self.verticalScrollBar().setValue(v_scroll)

        h_scroll = int(factor * self.horizontalScrollBar().value() +
                       ((factor-1) * self.horizontalScrollBar().pageStep()/2))
        self.horizontalScrollBar().setValue(h_scroll)

    def zoom_fit(self):
        """Zoom the displayed image to fit the available viewport."""

        image = self.image.pixmap().size()
        viewport = self.size()

        
        if image.height() == 0 or image.width() == 0:
            return

        v_zoom = viewport.height() / image.height()
        h_zoom = viewport.width() / image.width()
        self.zoom(min(v_zoom, h_zoom)*0.99, False)

    #This is a basic event handler which can be reimplemented in every widget class to receive wheel commands.
    def wheelEvent(self, event):  # pylint: disable=C0103
        """Handle scroll wheel events in the viewport.
        Scroll - Change current slice up or down."""
        
        slice_ = int(numpy.sign(event.delta()))
        self.dialog.refresh(slice_)
            
           