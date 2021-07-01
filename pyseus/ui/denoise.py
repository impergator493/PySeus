from PySide2.QtWidgets import QButtonGroup, QDesktopWidget, QDialog, QDialogButtonBox, QFormLayout, QLayout, QLineEdit, QMainWindow, QAction, QLabel, QFileDialog, \
                              QFrame, QPushButton, QRadioButton, QScrollArea, QSizePolicy, QVBoxLayout, QHBoxLayout, QWidget


from pyseus.denoising.threading_denoise import ThreadingDenoised
from pyseus.denoising.tv import TV
from pyseus.modes.grayscale import Grayscale
from PySide2.QtCore import Qt
import numpy
import scipy




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
        self.btn_all_slices = QRadioButton("Whole Dataset")
        self.grp_data_sel.addButton(self.btn_curr_slice,1)
        self.grp_data_sel.addButton(self.btn_all_slices,2)
        self.btn_all_slices_2D = QRadioButton("2D")
        self.btn_all_slices_3D = QRadioButton("3D")
        
        # subgroup of radio buttons to dataset selection
        self.lab_denoise_type = QLabel("Denoising Type")
        self.grp_tv_type = QButtonGroup()
        self.btn_tv_L1 = QRadioButton("L1")
        self.btn_tv_L1.setChecked(True)
        self.btn_tv_ROF = QRadioButton("HuberROF")
        self.grp_tv_type.addButton(self.btn_tv_L1, 1)
        self.grp_tv_type.addButton(self.btn_tv_ROF, 2)

        



        # form layout for parameter input for denoising algorithm
        # TODO change Names of rows to labels, so that alpha row can be hidden completely with the name of the row
        # not just the qlineedit field
        form = QFormLayout()
        self.qline_lambd = QLineEdit()
        #TV_lambda.setStyleSheet("color: white; background-color: darkgray")
        self.qline_lambd.setText("30")
        form.addRow("Lambda",self.qline_lambd)
        self.qline_iter = QLineEdit()
        self.qline_iter.setText("100")
        #TV_iter.setStyleSheet("color: white; background-color: darkgray")
        form.addRow("Iterations",self.qline_iter)
        self.qline_alpha = QLineEdit()
        self.qline_alpha.setText("0.03")
        size_pol = self.qline_alpha.sizePolicy()
        size_pol.setRetainSizeWhenHidden(True)
        self.qline_alpha.setSizePolicy(size_pol)
        self.qline_alpha.hide()
        #TV_alpha.setStyleSheet("color: white; background-color: darkgray")
        form.addRow("Alpha",self.qline_alpha)
        
        
        self.btn_tv_L1.clicked.connect(lambda: self.qline_alpha.hide())
        self.btn_tv_ROF.clicked.connect(lambda: self.qline_alpha.show())
        
        
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
        
        # @TODO check wether input types are okay
        alpha = float(self.qline_alpha.text())
        lambd = float(self.qline_lambd.text())
        iterations = int(self.qline_iter.text())  

        whole_data = None
        btn_data_id = self.grp_data_sel.checkedId()
        if btn_data_id == 1:
            whole_data = False
        elif btn_data_id == 2:
            whole_data = True

        tv_type = self.grp_tv_type.checkedId()

        self.window_denoised.open_window(alpha,lambd,iterations,whole_data,tv_type)




class DenoisedWindow(QDialog):

    def __init__(self,app):
        super().__init__()

        self.app = app
        self.denoised = None
        self.slice_id_selected = None
        self.whole_dataset = None

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

    
    # def denoised_callback(self, data):
        
    #     self.denoised = data

    #     # can it be done with that variable? It must not be changed under any circumstances 
    #     # otherwhise it is not consistent with the calculation! Lock dialog during calculation?

    #     # if data.ndim == 3?
    #     if self.whole_dataset:

    #         self.slice_id_selected = (self.app.dataset.slice_count() // 2)
    #         denoised_displayed = data[self.slice_id_selected,:,:]

    #     else:

    #         denoised_displayed = data

    #     self.display_image(denoised_displayed)

       
    

    def open_window(self,alpha,lambd,iterations,whole_data, tv_type):

         #print(self.grp_tv_type.checkedId())

        denoise = TV()
        
        self.whole_dataset = whole_data

        # self.denoised = numpy.zeros(noisy.shape)
        # denoise_thread = ThreadingDenoised(self.denoised, denoise.tv_denoising_L2, self.cb, (noisy, lambd, iterations))
        # denoise_thread.run()



        #noisy = scipy.io.loadmat('./tests/cameraman_noise.mat')['im']

        if self.whole_dataset:
            noisy = self.app.dataset.get_pixeldata(-1)
            
            # @TODO generalize TV Class for acceptance of 3D or create own 3D method that calls 2D method?
            # then there is no need anymore for 2 times  if tv_type case

            #L1 needs a small lambda for denoising, better can be seen with saltn pepper noise of cameraman standard noised pic

            self.denoised = numpy.zeros(noisy.shape)
            for i in range(0, self.app.dataset.slice_count()):
                if tv_type == 1:
                    self.denoised[i,:,:] = denoise.tv_denoising_L1(noisy[i,:,:],lambd,iterations)
                if tv_type == 2:
                    self.denoised[i,:,:] = denoise.tv_denoising_huberROF(noisy[i,:,:],lambd,iterations,alpha)


            self.slice_id_selected = (self.app.dataset.slice_count() // 2)
            denoised_displayed = self.denoised[self.slice_id_selected,:,:]


        
        else:    
            noisy = self.app.dataset.get_pixeldata(self.app.get_slice_id())
            
            if tv_type == 1:
                self.denoised = denoise.tv_denoising_L1(noisy,lambd,iterations)
            if tv_type == 2:
                self.denoised = denoise.tv_denoising_huberROF(noisy,lambd,iterations,alpha)
            denoised_displayed = self.denoised
        
          
        self.display_image(denoised_displayed)
       
        
     

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
        
        if self.whole_dataset:
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


        
            
           