from PySide2.QtWidgets import QBoxLayout, QButtonGroup, QCheckBox, QDesktopWidget, QDialog, QDialogButtonBox, QFormLayout, QGroupBox, QLayout, QLineEdit, QMainWindow, QAction, QLabel, QFileDialog, \
                              QFrame, QPushButton, QRadioButton, QScrollArea, QSizePolicy, QVBoxLayout, QHBoxLayout, QWidget


from ..settings import ProcessType, ProcessSelDataType, ProcessRegType, DataType

from pyseus.processing.threading_process import ProcessThread
from pyseus.processing.tv import TV
from pyseus.processing.tgv import TGV
from pyseus.processing.tgv_reconstruction import TGV_Reco
from pyseus.modes.grayscale import Grayscale
from PySide2.QtCore import Qt
import numpy
import scipy



class ProcessDialog(QDialog):
    """ Dialog for Image Processing with input parameters "denoising" or "reconstruction" as processing type"""

    def __init__(self,app, proc_type):
        super().__init__()
  
        self.app = app
        self.proc_type = proc_type
        self.window_processed = ProcessedWindow(app, self.proc_type)
        

        vlayout_sel_par = QVBoxLayout()
        vlayout_sel = QVBoxLayout()
        vlayout_type = QVBoxLayout()
        grp_box_sel = QGroupBox("Data Selection")
        grp_box_type = QGroupBox("Model Type")

        hlayout = QHBoxLayout()
        # Take "Denoising" or "Reconstruction" as title straight from the ENUM
        self.setWindowTitle(str.capitalize(self.proc_type.name))
        
        # subgroup of radio buttons for data selection
        self.grp_data_sel = QButtonGroup()
        self.btn_curr_slice = QRadioButton("Current Slice")
        self.btn_curr_slice.setChecked(True)
        self.btn_all_slices_2D = QRadioButton("2D - Whole Scan")
        self.btn_all_slices_3D = QRadioButton("3D - Whole Scan")
        self.grp_data_sel.addButton(self.btn_curr_slice,ProcessSelDataType.SLICE_2D)
        self.grp_data_sel.addButton(self.btn_all_slices_2D,ProcessSelDataType.WHOLE_SCAN_2D)
        self.grp_data_sel.addButton(self.btn_all_slices_3D,ProcessSelDataType.WHOLE_SCAN_3D)
        self.chbx_coil = QCheckBox("Use coil sensitivities")
        self.chbx_spmask = QCheckBox("Use undersampling sparse mask")

        
        # subgroup of radio buttons to dataset selection
        self.grp_tv_type = QButtonGroup()
        self.btn_tv_L1 = QRadioButton("TV-L1")
        self.btn_tv_L1.setChecked(True)
        self.btn_hub_L2 = QRadioButton("Huber-L2")
        self.btn_tv_L2 = QRadioButton("TV-L2")
        self.btn_tgv2_L2 = QRadioButton("TGV2-L2")
        self.grp_tv_type.addButton(self.btn_tv_L1, int(ProcessRegType.TV_L1))
        self.grp_tv_type.addButton(self.btn_hub_L2, int(ProcessRegType.HUB_L2))
        self.grp_tv_type.addButton(self.btn_tv_L2, int(ProcessRegType.TV_L2))
        self.grp_tv_type.addButton(self.btn_tgv2_L2, int(ProcessRegType.TGV2_L2))
        
        # form layout for parameter input for processing algorithm
        
        vlayout_par = QVBoxLayout()
        grp_box_par = QGroupBox("Parameters")

        v_form1 = QFormLayout()
        self.qline_lambd = QLineEdit()
        self.qline_lambd.setText("30")
        v_form1.addRow("Lambda",self.qline_lambd)
        
        self.qline_iter = QLineEdit()
        self.qline_iter.setText("100")
        v_form1.addRow("Iterations",self.qline_iter)

        v_form1.addRow(" ", None)
        self.qline_alpha = QLineEdit()
        self.qline_alpha.setText("0.03")
        size_pol = self.qline_alpha.sizePolicy()
        size_pol.setRetainSizeWhenHidden(True)
        self.qline_alpha.setSizePolicy(size_pol)
        self.qline_alpha.hide()
        v_form1.addRow("Alpha",self.qline_alpha)

        self.qline_alpha0 = QLineEdit()
        self.qline_alpha0.setText("2")
        size_pol0 = self.qline_alpha0.sizePolicy()
        size_pol0.setRetainSizeWhenHidden(True)
        self.qline_alpha0.setSizePolicy(size_pol0)
        self.qline_alpha0.hide()
        v_form1.addRow("Alpha0",self.qline_alpha0)

        self.qline_alpha1 = QLineEdit()
        self.qline_alpha1.setText("1")
        size_pol1 = self.qline_alpha1.sizePolicy()
        size_pol1.setRetainSizeWhenHidden(True)
        self.qline_alpha1.setSizePolicy(size_pol1)
        self.qline_alpha1.hide()
        v_form1.addRow("Alpha1",self.qline_alpha1)
        
        self.btn_tv_L1.clicked.connect(lambda: self.qline_alpha.hide())
        self.btn_hub_L2.clicked.connect(lambda: self.qline_alpha.show())
        self.btn_tv_L2.clicked.connect(lambda: self.qline_alpha.hide())
        self.btn_tgv2_L2.clicked.connect(lambda: self.qline_alpha.hide())

        self.btn_tv_L1.clicked.connect(lambda: self.qline_alpha0.hide())
        self.btn_hub_L2.clicked.connect(lambda: self.qline_alpha0.hide())
        self.btn_tv_L2.clicked.connect(lambda: self.qline_alpha0.hide())
        self.btn_tgv2_L2.clicked.connect(lambda: self.qline_alpha0.show())

        self.btn_tv_L1.clicked.connect(lambda: self.qline_alpha1.hide())
        self.btn_hub_L2.clicked.connect(lambda: self.qline_alpha1.hide())
        self.btn_tv_L2.clicked.connect(lambda: self.qline_alpha1.hide())
        self.btn_tgv2_L2.clicked.connect(lambda: self.qline_alpha1.show())
        
        
        # function without brackets just connects the function, but does not call it
        self.box_btns = QDialogButtonBox()
        self.box_btns.setStandardButtons(QDialogButtonBox.Cancel | QDialogButtonBox.Ok)
        self.box_btns.accepted.connect(self.signal_ok)
        self.box_btns.rejected.connect(lambda:self.close())
   

        # organize items on GUI
        vlayout_sel.addWidget(self.btn_curr_slice)
        vlayout_sel.addWidget(self.btn_all_slices_2D)
        vlayout_sel.addWidget(self.btn_all_slices_3D)
        if self.proc_type == ProcessType.RECONSTRUCTION:
            vlayout_sel.addWidget(self.chbx_coil)
            vlayout_sel.addWidget(self.chbx_spmask)
        grp_box_sel.setLayout(vlayout_sel)

        vlayout_type.addWidget(self.btn_tv_L1)
        vlayout_type.addWidget(self.btn_hub_L2)
        vlayout_type.addWidget(self.btn_tv_L2)
        vlayout_type.addWidget(self.btn_tgv2_L2)
        grp_box_type.setLayout(vlayout_type)

        vlayout_sel_par.addWidget(grp_box_sel)
        vlayout_sel_par.addWidget(grp_box_type)

        vlayout_par.addLayout(v_form1)
        vlayout_par.addWidget(self.box_btns)
        grp_box_par.setLayout(vlayout_par)

        hlayout.addLayout(vlayout_sel_par)
        hlayout.addWidget(grp_box_par)
        
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
                            "QGroupBox"
                                    "{"
                                    "color: white"
                                    "}"
                            "QCheckBox"
                                    "{"
                                    "color: white"
                                    "}"
                                )


    def signal_ok(self):
        
        # @TODO check wether input types are okay
        alpha = float(self.qline_alpha.text())
        alpha0 = float(self.qline_alpha0.text())
        alpha1 = float(self.qline_alpha1.text())
        lambd = float(self.qline_lambd.text())
        iterations = int(self.qline_iter.text())

        
        #according to definition in init method, 1 = 2D, 2 = 2D - whole dataset, 3 = 3D - whole Dataset
        dataset_type = ProcessSelDataType(self.grp_data_sel.checkedId())
        tv_type = ProcessRegType(self.grp_tv_type.checkedId())

        use_coilmap = self.chbx_coil.isChecked()
        use_spmask = self.chbx_spmask.isChecked()

        self.window_processed.start_calculation(alpha, alpha0, alpha1, lambd, iterations,dataset_type,tv_type, use_coilmap, use_spmask)




class ProcessedWindow(QDialog):

    def __init__(self,app, proc_type):
        super().__init__()

        self.app = app
        self.processed = None
        self.array_shape = None
        self.slice_id_selected = None
        self.dataset_type = None
        self.proc_type = proc_type

        self.setWindowTitle("Processed Data")

        self.view = ProcessedViewWidget(self.app, self)
        self.box_btns_ok = QDialogButtonBox()
        self.box_btns_ok.setStandardButtons(QDialogButtonBox.Cancel | QDialogButtonBox.Ok)
        self.box_btns_ok.accepted.connect(self.signal_ok)
        self.box_btns_ok.rejected.connect(lambda:self.close())

        self.setLayout(QVBoxLayout())
        self.layout().addWidget(self.view)
        self.layout().addWidget(self.box_btns_ok)


        self.mode = Grayscale()


    def calculation_callback(self,data_obj):
        
        
        self.processed = data_obj

        #TODO Remove fixed slice id
        if self.dataset_type == ProcessSelDataType.WHOLE_SCAN_2D or self.dataset_type == ProcessSelDataType.WHOLE_SCAN_3D :
            #self.slice_id_selected = (self.app.dataset.slice_count() // 2)
            self.slice_id_selected = 3
            processed_displayed = self.processed[self.slice_id_selected,:,:]
        elif self.dataset_type == ProcessSelDataType.SLICE_2D:
            processed_displayed = self.processed

        self.display_image(processed_displayed)

       
    

    def start_calculation(self,alpha, alpha0, alpha1, lambd,iterations,dataset_type, tv_type, use_coilmap, use_spmask):

        self.dataset_type = dataset_type

        if self.proc_type == ProcessType.DENOISING:
            
            if self.dataset_type == ProcessSelDataType.WHOLE_SCAN_2D or self.dataset_type == ProcessSelDataType.WHOLE_SCAN_3D:
                dataset_noisy = self.app.dataset.get_pixeldata(-1)
            elif self.dataset_type == ProcessSelDataType.SLICE_2D:
                dataset_noisy = self.app.dataset.get_pixeldata(self.app.get_slice_id())

            if tv_type == ProcessRegType.TV_L1:
                tv_class = TV()
                tv_type_func = tv_class.tv_denoising_L1
                params = (lambd, iterations)
            if tv_type == ProcessRegType.HUB_L2:
                tv_class = TV()
                tv_type_func = tv_class.tv_denoising_huberROF
                params = (lambd, iterations, alpha)
            if tv_type == ProcessRegType.TV_L2:
                tv_class = TV()
                tv_type_func = tv_class.tv_denoising_L2
                params = (lambd, iterations)
            if tv_type == ProcessRegType.TGV2_L2:
                #tv_type_func not needed, just one possible case for tgv
                tv_class = TGV()
                tv_type_func = None
                params = (lambd, alpha0, alpha1, iterations)

            thread_processed = ProcessThread(self, tv_class, tv_type_func, dataset_type, dataset_noisy, params)
            thread_processed.output.connect(self.calculation_callback)
            thread_processed.start()
        
        elif self.proc_type == ProcessType.RECONSTRUCTION:
            
            if self.app.dataset.get_data_type() != DataType.KSPACE:
                raise TypeError("Loaded dataset must be kspace data")

            else:

                scan_id = self.app.dataset.scan
                slice_id = self.app.get_slice_id()
                #@TODO: Remove selection for specific Slices again, just to see if it works in general
                if self.dataset_type == ProcessSelDataType.WHOLE_SCAN_2D or self.dataset_type == ProcessSelDataType.WHOLE_SCAN_3D:
                    dataset_kspace = self.app.dataset.get_reco_pixeldata(scan_id, -1)[:,69:75,:,:]#dataset_noisy = self.app.dataset.get_pixeldata(-1)
                elif self.dataset_type == ProcessSelDataType.SLICE_2D:
                    dataset_kspace = self.app.dataset.get_reco_pixeldata(scan_id, slice_id)#dataset_noisy = self.app.dataset.get_pixeldata(self.app.get_slice_id())

                sparse_mask = numpy.ones_like(dataset_kspace)
                data_coils = numpy.ones_like(dataset_kspace)

                # if no sparse mask given, sample is fully sampled and the whole array is 1
                if use_spmask:
                    pass
                    # add function to import sparse mask or to create one
                # if no sensitivity map is given, the whole array contains 1
                if use_coilmap:
                    if self.dataset_type == ProcessSelDataType.WHOLE_SCAN_2D or self.dataset_type == ProcessSelDataType.WHOLE_SCAN_3D:
                        #TODO Remove selection for specific slices later again
                        data_coils = self.app.dataset.get_coil_data(-1)[:,69:75,:,:]
                    elif self.dataset_type == ProcessSelDataType.SLICE_2D:
                        data_coils = self.app.dataset.get_coil_data(slice_id)

                #if tv_type == ProcessRegType.TV_L1:
                    #tv_class = TV()
                    #tv_type_func = tv_class.tv_denoising_L1
                    #params = (lambd, iterations)
                #if tv_type == ProcessRegType.TV_ROF:
                    #tv_class = TV()
                    #tv_type_func = tv_class.tv_denoising_huberROF
                    #params = (lambd, iterations, alpha)
                #if tv_type == ProcessRegType.TV_L2:
                    #tv_class = TV()
                    #tv_type_func = tv_class.tv_denoising_L2
                    #params = (lambd, iterations)
                if tv_type == ProcessRegType.TGV2_L2:
                    #tv_type_func not needed, just one possible case for tgv
                    tv_class = TGV_Reco()
                    tv_type_func = None
                    params = (lambd, alpha0, alpha1, iterations)

                    # zum probieren um debuggen zu können temporär:
                    self.calculation_callback(tv_class.tgv2_reconstruction_gen(dataset_type, dataset_kspace, 
                                                                                data_coils, sparse_mask, *params))

                # thread_processed = ProcessThread(self, tv_class, tv_type_func, dataset_type, dataset_kspace, params, sparse_mask, data_coils)
                # thread_processed.output.connect(self.calculation_callback)
                # thread_processed.start()

        # Threading
        # should be done with .start() method, not with run
        # otherwhise threading wont be activated
        
        
        
     

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
        
        if self.dataset_type == ProcessSelDataType.WHOLE_SCAN_2D or self.dataset_type == ProcessSelDataType.WHOLE_SCAN_3D:
            new_slice = self.slice_id_selected + slice_inc
            if 0 <= new_slice < self.app.dataset.slice_count():
                self.display_image(self.processed[new_slice])
                self.slice_id_selected = new_slice
            elif new_slice < 0:
                self.slice_id_selected = 0
            else:
                self.slice_id_selected = self.app.dataset.slice_count()

    def signal_ok(self):
        
        self.app.set_processed_dataset(self.processed)

    def resizeEvent(self, event):  # pylint: disable=C0103
        """Keep the viewport centered and adjust zoom on window resize."""
        x_factor = event.size().width() / event.oldSize().width()
        # y_factor = event.size().height() / event.oldSize().height()
        # @TODO x_factor if xf < yf or xf * width * zoom_factor < viewport_x
        self.view.zoom(x_factor, True)
        
        

class ProcessedViewWidget(QScrollArea):
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


        
            
           