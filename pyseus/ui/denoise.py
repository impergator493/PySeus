from PySide2.QtWidgets import QButtonGroup, QDesktopWidget, QDialog, QDialogButtonBox, QFormLayout, QLayout, QLineEdit, QMainWindow, QAction, QLabel, QFileDialog, \
                              QFrame, QPushButton, QRadioButton, QScrollArea, QSizePolicy, QVBoxLayout, QHBoxLayout, QWidget


from pyseus.denoising.threading_denoise import ThreadingDenoised
from pyseus.denoising.tv import TV
#from pyseus.denoising.tgv import TGV
from pyseus.denoising.tgv import TGV
#from pyseus.denoising.tgv_3D import TGV_3D
from pyseus.modes.grayscale import Grayscale
from PySide2.QtCore import Qt
import numpy
import scipy

import matplotlib.pyplot as plt


class DenoiseDialog(QDialog):

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
        self.btn_all_slices_2D = QRadioButton("2D - Whole Dataset")
        self.btn_all_slices_3D = QRadioButton("3D - Whole Dataset")
        self.grp_data_sel.addButton(self.btn_curr_slice,1)
        self.grp_data_sel.addButton(self.btn_all_slices_2D,2)
        self.grp_data_sel.addButton(self.btn_all_slices_3D,3)

        
        # subgroup of radio buttons to dataset selection
        self.lab_denoise_type = QLabel("Denoising Type")
        self.grp_tv_type = QButtonGroup()
        self.btn_tv_L1 = QRadioButton("L1")
        self.btn_tv_L1.setChecked(True)
        self.btn_tv_ROF = QRadioButton("HuberROF")
        self.btn_tv_L2 = QRadioButton("L2")
        self.btn_tgv2 = QRadioButton("TGV2")
        self.grp_tv_type.addButton(self.btn_tv_L1, 1)
        self.grp_tv_type.addButton(self.btn_tv_ROF, 2)
        self.grp_tv_type.addButton(self.btn_tv_L2, 3)
        self.grp_tv_type.addButton(self.btn_tgv2, 4)

        
        # form layout for parameter input for denoising algorithm
        form = QFormLayout()
        self.qline_lambd = QLineEdit()
        self.qline_lambd.setText("30")
        form.addRow("Lambda",self.qline_lambd)
        
        self.qline_iter = QLineEdit()
        self.qline_iter.setText("100")
        form.addRow("Iterations",self.qline_iter)

        self.qline_alpha = QLineEdit()
        self.qline_alpha.setText("0.03")
        size_pol = self.qline_alpha.sizePolicy()
        size_pol.setRetainSizeWhenHidden(True)
        self.qline_alpha.setSizePolicy(size_pol)
        self.qline_alpha.hide()
        form.addRow("Alpha",self.qline_alpha)

        self.qline_alpha0 = QLineEdit()
        self.qline_alpha0.setText("0.5")
        size_pol0 = self.qline_alpha0.sizePolicy()
        size_pol0.setRetainSizeWhenHidden(True)
        self.qline_alpha0.setSizePolicy(size_pol0)
        self.qline_alpha0.hide()
        form.addRow("Alpha0",self.qline_alpha0)

        self.qline_alpha1 = QLineEdit()
        self.qline_alpha1.setText("0.5")
        size_pol1 = self.qline_alpha1.sizePolicy()
        size_pol1.setRetainSizeWhenHidden(True)
        self.qline_alpha1.setSizePolicy(size_pol1)
        self.qline_alpha1.hide()
        form.addRow("Alpha1",self.qline_alpha1)
        
        self.btn_tv_L1.clicked.connect(lambda: self.qline_alpha.hide())
        self.btn_tv_ROF.clicked.connect(lambda: self.qline_alpha.show())
        self.btn_tv_L2.clicked.connect(lambda: self.qline_alpha.hide())
        self.btn_tgv2.clicked.connect(lambda: self.qline_alpha.hide())

        self.btn_tv_L1.clicked.connect(lambda: self.qline_alpha0.hide())
        self.btn_tv_ROF.clicked.connect(lambda: self.qline_alpha0.hide())
        self.btn_tv_L2.clicked.connect(lambda: self.qline_alpha0.hide())
        self.btn_tgv2.clicked.connect(lambda: self.qline_alpha0.show())

        self.btn_tv_L1.clicked.connect(lambda: self.qline_alpha1.hide())
        self.btn_tv_ROF.clicked.connect(lambda: self.qline_alpha1.hide())
        self.btn_tv_L2.clicked.connect(lambda: self.qline_alpha1.hide())
        self.btn_tgv2.clicked.connect(lambda: self.qline_alpha1.show())
        
        
        # function without brackets just connects the function, but does not call it
        self.box_btns = QDialogButtonBox()
        self.box_btns.setStandardButtons(QDialogButtonBox.Cancel | QDialogButtonBox.Ok)
        self.box_btns.accepted.connect(self.signal_ok)
        self.box_btns.rejected.connect(lambda:self.close())
   

        # organize items on GUI
        vlayout.addWidget(self.lab_data_sel)
        vlayout.addWidget(self.btn_curr_slice)
        vlayout.addWidget(self.btn_all_slices_2D)
        vlayout.addWidget(self.btn_all_slices_3D)
        vlayout.addWidget(self.lab_denoise_type)
        vlayout.addWidget(self.btn_tv_L1)
        vlayout.addWidget(self.btn_tv_ROF)
        vlayout.addWidget(self.btn_tv_L2)
        vlayout.addWidget(self.btn_tgv2)
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
        
        # @TODO check wether input types are okay
        alpha = float(self.qline_alpha.text())
        alpha0 = float(self.qline_alpha0.text())
        alpha1 = float(self.qline_alpha1.text())
        lambd = float(self.qline_lambd.text())
        iterations = int(self.qline_iter.text())  

        btn_data_id = self.grp_data_sel.checkedId()
        
        #according to definition in init method, 1 = 2D, 2 = 2D - whole dataset, 3 = 3D - whole Dataset
        dataset_type = btn_data_id

        tv_type = self.grp_tv_type.checkedId()

        self.window_denoised.open_window(alpha, alpha0, alpha1, lambd, iterations,dataset_type,tv_type)




class DenoisedWindow(QDialog):

    def __init__(self,app):
        super().__init__()

        self.app = app
        self.denoised = None
        self.array_shape = None
        self.slice_id_selected = None
        self.dataset_type = None


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

    # the problem is, that here still the thread is a seprated one and therefor and information
    # cannot be shown graphically (matplotlib doesnt work either)
    def denoised_callback(self,data_obj):
        
        
        self.denoised = data_obj

        
        if self.dataset_type == 2 or self.dataset_type == 3:
            self.slice_id_selected = (self.app.dataset.slice_count() // 2)
            denoised_displayed = self.denoised[self.slice_id_selected,:,:]
        elif self.dataset_type == 1:
            denoised_displayed = self.denoised

        self.display_image(denoised_displayed)

       
    

    def open_window(self,alpha, alpha0, alpha1, lambd,iterations,dataset_type, tv_type):

         #print(self.grp_tv_type.checkedId())

        
        self.dataset_type = dataset_type

        #noisy = scipy.io.loadmat('./tests/cameraman_noise.mat')['im']

        if self.dataset_type == 2 or self.dataset_type == 3:
            dataset_noisy = self.app.dataset.get_pixeldata(-1)
        elif self.dataset_type == 1:
            dataset_noisy = self.app.dataset.get_pixeldata(self.app.get_slice_id())

            #L1 needs a small lambda for denoising, better can be seen with saltn pepper noise of cameraman standard noised pic
        
        if tv_type == 1:
            tv_class = TV()
            tv_type_func = tv_class.tv_denoising_L1
            params = (lambd, iterations)
        if tv_type == 2:
            tv_class = TV()
            tv_type_func = tv_class.tv_denoising_huberROF
            params = (lambd, iterations, alpha)
        if tv_type == 3:
            tv_class = TV()
            tv_type_func = tv_class.tv_denoising_L2
            params = (lambd, iterations)
        if tv_type == 4:
            #tv_type_func not needed, just one possible case for tgv
            tv_class = TGV()
            tv_type_func = None
            params = (alpha0, alpha1, iterations)
        
        # should be done with .start() method, not with run
        # otherwhise threading wont be activated
        
        thread_denoised = ThreadingDenoised(self, tv_class, tv_type_func, dataset_type, dataset_noisy, params)
        thread_denoised.output.connect(self.denoised_callback)
        thread_denoised.start()
        
     

    def display_image(self, image):
        

        # @TODO
        # shortcut, thats not a good solution
        # should the original grayscale object be used with temporary window
        # or a new grayscale objecte be generated which is independent?
        self.mode.temporary_window(image)
        pixmap = self.mode.get_pixmap(image)
        self.view.set(pixmap)
        #self.setGeometry(600,300,600,600)
        self.show()
        screen_size = QDesktopWidget().screenGeometry()
        self.resize(screen_size.width()*0.3, screen_size.height()*0.3)
        self.view.zoom_fit()

     
    def refresh_slice(self, slice_inc):
        
        if self.dataset_type == 2 or self.dataset_type == 3:
            new_slice = self.slice_id_selected + slice_inc
            if 0 <= new_slice < self.app.dataset.slice_count():
                self.display_image(self.denoised[new_slice])
                self.slice_id_selected = new_slice
            elif new_slice < 0:
                self.slice_id_selected = 0
            else:
                self.slice_id_selected = self.app.dataset.slice_count()

    def signal_ok(self):
        
        self.app.set_denoised_dataset(self.denoised)

    def resizeEvent(self, event):  # pylint: disable=C0103
        """Keep the viewport centered and adjust zoom on window resize."""
        x_factor = event.size().width() / event.oldSize().width()
        # y_factor = event.size().height() / event.oldSize().height()
        # @TODO x_factor if xf < yf or xf * width * zoom_factor < viewport_x
        self.view.zoom(x_factor, True)
        
        

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
        self.dialog.refresh_slice(slice_)


        
            
           