"""Main window for PySeus.

Classes
-------

**MainWindow** - Class representing the main window for PySeus.
"""

import sys
import webbrowser
from functools import partial
import os
from PySide2 import QtWidgets
from PySide2.QtCore import QLine

from PySide2.QtWidgets import QButtonGroup, QDialog, QDialogButtonBox, QFormLayout, QLineEdit, QMainWindow, QAction, QLabel, QFileDialog, \
                              QFrame, QPushButton, QRadioButton, QVBoxLayout, QHBoxLayout, QWidget
from PySide2.QtGui import QIcon

from .view import ViewWidget
from .sidebar import ConsoleWidget, InfoWidget, MetaWidget
from .thumbs import ThumbsWidget

from pyseus.denoising.tv import TV
import scipy.io
import matplotlib.pyplot as plt




class MainWindow(QMainWindow):  # pylint: disable=R0902
    """Class representing the main window for PySeus."""

    def __init__(self, app):
        QMainWindow.__init__(self)
        self.setWindowTitle("PySEUS")

        self.app = app
        """Reference to the main application object."""

        self.thumbs = ThumbsWidget(app)
        """Reference to the thumbs widget."""

        self.view = ViewWidget(app)
        """Reference to the view widget."""

        self.info = InfoWidget(app)
        """Reference to the info sidebar widget."""

        self.meta = MetaWidget(app)
        """Reference to the meta sidebar widget."""

        self.console = ConsoleWidget(app)
        """Reference to the console sidebar widget."""

        # Default path for file open dialoge
        self._open_path = ""

        # Horizontal layout (thumbs, view, sidebar)
        wrapper = QFrame(self)
        wrapper.setLayout(QHBoxLayout())
        wrapper.layout().setContentsMargins(0, 0, 0, 0)
        wrapper.layout().addWidget(self.thumbs)
        wrapper.layout().addWidget(self.view)

        # Sidebar / Vertical layout (info, meta, console)
        sidebar = QFrame(self)
        sidebar.setLayout(QVBoxLayout())
        sidebar.layout().setContentsMargins(0, 0, 5, 0)

        sidebar.layout().addWidget(SidebarHeading("File Info", True))
        sidebar.layout().addWidget(self.info)

        sidebar.layout().addWidget(SidebarHeading("Metadata"))
        sidebar.layout().addWidget(self.meta)

        sidebar.layout().addWidget(SidebarHeading("Console"))
        sidebar.layout().addWidget(self.console)

        wrapper.layout().addWidget(sidebar)

        self.setup_menu()

        self.statusBar().setSizeGripEnabled(False)

        self.setCentralWidget(wrapper)

        icon = QIcon(os.path.abspath(os.path.join(
            os.path.dirname(__file__), "./icon.png")))
        self.setWindowIcon(icon)

        # Window dimensions
        geometry = self.app.qt_app.desktop().availableGeometry(self)
        self.resize(geometry.width() * 0.6, geometry.height() * 0.6)


    def add_menu_item(self, menu, title, callback, shortcut=""):
        """Create a menu item."""
        action = QAction(title, self)
        if shortcut != "":
            action.setShortcut(shortcut)
        action.triggered.connect(callback)
        menu.addAction(action)
        return action

    def setup_menu(self):
        """Setup the menu bar. Items in the *Evaluate* menu are created
        in the *setup_menu* function of tool classes."""
        ami = self.add_menu_item
        menu_bar = self.menuBar()

        self.file_menu = menu_bar.addMenu("&File")
        ami(self.file_menu, "&Load", self._action_open, "Ctrl+O")
        ami(self.file_menu, "&Reload", self._action_reload, "Ctrl+L")
        self.file_menu.addSeparator()
        ami(self.file_menu, "&Quit", self._action_quit, "Ctrl+Q")

        self.view_menu = menu_bar.addMenu("&View")
        for mode in self.app.modes:
            mode.setup_menu(self.app, self.view_menu, self.add_menu_item)
        self.view_menu.addSeparator()
        
        ami(self.view_menu, "Zoom &in", self._action_zoom_in, "+")
        ami(self.view_menu, "Zoom &out", self._action_zoom_out, "-")
        ami(self.view_menu, "Zoom to &fit", self._action_zoom_fit, "#")
        ami(self.view_menu, "Reset &Zoom", self._action_zoom_reset, "0")
        self.view_menu.addSeparator()

        ami(self.view_menu, "&Lower Window", self._action_win_lower, "q")
        ami(self.view_menu, "&Raise Window", self._action_win_raise, "w")
        ami(self.view_menu, "&Shrink Window", self._action_win_shrink, "a")
        ami(self.view_menu, "&Enlarge Window", self._action_win_enlarge, "s")
        ami(self.view_menu, "Reset &Window", self._action_win_reset, "d")

        self.explore_menu = menu_bar.addMenu("E&xplore")
        ami(self.explore_menu, "Nex&t Slice",
            partial(self._action_slice, 1), "PgUp")
        ami(self.explore_menu, "P&revious Slice",
            partial(self._action_slice, -1), "PgDown")
        self.explore_menu.addSeparator()

        ami(self.explore_menu, "Rotate z",
            partial(self._action_rotate, 2), "Ctrl+E")
        ami(self.explore_menu, "Rotate x",
            partial(self._action_rotate, 1), "Ctrl+R")
        ami(self.explore_menu, "Rotate y",
            partial(self._action_rotate, 0), "Ctrl+T")
        self.explore_menu.addSeparator()
        ami(self.explore_menu, "Flip x (L-R)",
            partial(self._action_flip, 1), "Ctrl+D")
        ami(self.explore_menu, "Flip y (U-D)",
            partial(self._action_flip, 0), "Ctrl+F")
        ami(self.explore_menu, "Flip z (F-B)",
            partial(self._action_flip, 2), "Ctrl+G")
        self.explore_menu.addSeparator()
        ami(self.explore_menu, "Reset Scan",
            partial(self._action_rotate, -1), "Ctrl+Z")
        self.explore_menu.addSeparator()

        ami(self.explore_menu, "Next &Scan",
            partial(self._action_scan, 1), "Alt+PgUp")
        ami(self.explore_menu, "Previous Sc&an",
            partial(self._action_scan, -1), "Alt+PgDown")
        self.explore_menu.addSeparator()
        ami(self.explore_menu, "Cine Play", self._action_cine, "Ctrl+#")

        self.tools_menu = menu_bar.addMenu("&Evaluate")
        for tool in self.app.tools:
            tool.setup_menu(self.app, self.tools_menu, self.add_menu_item)
        self.tools_menu.addSeparator()
        ami(self.tools_menu, "&Clear RoI", self._action_tool_clear, "Esc")

        self.denoise_menu = menu_bar.addMenu("&Denoise")
        ami(self.denoise_menu, "TV",
            self._open_dialog_tv, "---")
        ami(self.denoise_menu, "TGV",
            print('test'), "---")
        ami(self.denoise_menu, "H1",
            print('test'), "---")


        # About action is its own top level menu
        ami(menu_bar, "&About", self._action_about)

    def show_status(self, message):
        """Display *message* in the status bar."""
        self.statusBar().showMessage(message)

    def resizeEvent(self, event):  # pylint: disable=C0103
        """Keep the viewport centered and adjust zoom on window resize."""
        x_factor = event.size().width() / event.oldSize().width()
        # y_factor = event.size().height() / event.oldSize().height()
        # @TODO x_factor if xf < yf or xf * width * zoom_factor < viewport_x
        self.view.zoom(x_factor, True)

    def _action_quit(self):
        self.app.qt_app.quit()
        sys.exit()

    def _action_open(self):
        path, _ = QFileDialog.getOpenFileName(None, "Open file",
                                              self._open_path, "*.*")

        if not path == "":
            self._open_path = os.path.dirname(path)
            self.app.load_file(path)

    def _action_reload(self):
        if self.app.dataset is not None:
            self.app.load_file(self.app.dataset.path)

    def _action_zoom_in(self):
        self.view.zoom(1.25)

    def _action_zoom_out(self):
        self.view.zoom(0.8)

    def _action_zoom_fit(self):
        self.view.zoom_fit()

    def _action_zoom_reset(self):
        self.view.zoom(1, False)

    def _action_about(self):  # pylint: disable=R0201
        webbrowser.open_new("https://github.com/IMTtugraz/PySeus")

    def _action_win_lower(self):
        self.app.mode.move_window(-20)
        self.app.refresh()

    def _action_win_raise(self):
        self.app.mode.move_window(20)
        self.app.refresh()

    def _action_win_shrink(self):
        self.app.mode.scale_window(-25)
        self.app.refresh()

    def _action_win_enlarge(self):
        self.app.mode.scale_window(25)
        self.app.refresh()

    def _action_win_reset(self):
        self.app.mode.reset_window()
        self.app.refresh()

    def _action_slice(self, step):
        self.app.select_slice(step, True)

    def _action_scan(self, step):
        self.app.select_scan(step, True)

    def _action_tool_clear(self):
        self.app.clear_tool()

    def _action_rotate(self, axis):
        self.app.rotate(axis)

    def _action_flip(self, direction):
        self.app.flip(direction)

    def _action_cine(self):
        self.app.toggle_cine()

    def _open_dialog_tv(self):
    
        self.alpha = 0
        self.lambd = 0
        self.iter = 0
        

        dialog = QDialog()
        vlayout = QVBoxLayout()
        hlayout = QHBoxLayout()
        dialog.setWindowTitle("Denoise")
        
        data_selection = QLabel("Data Selection")
        slice_group = QButtonGroup()
        curr_slice = QRadioButton("Current Slice")
        curr_slice.setChecked(True)
        whole_dataset = QRadioButton("Whole Dataset")
        slice_group.addButton(curr_slice)
        slice_group.addButton(whole_dataset)
        whole_dataset_2D = QRadioButton("2D")
        whole_dataset_3D = QRadioButton("3D")

        # subgroup of radio buttons to dataset selection
        denoise_type = QLabel("Denoising Type")

        tv_group = QButtonGroup()
        tv_L1 = QRadioButton("L1")
        tv_L2 = QRadioButton("L2")
        tv_ROF = QRadioButton("HuberROF")
        tv_group.addButton(tv_L1, 1)
        tv_group.addButton(tv_L2, 2)
        tv_group.addButton(tv_ROF, 3)

        self.tv_group = tv_group

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
        btns = QDialogButtonBox()
        btns.setStandardButtons(QDialogButtonBox.Cancel | QDialogButtonBox.Ok)
        btns.accepted.connect(self.signal_ok)
        btns.rejected.connect(lambda:dialog.close())
   
        

        vlayout.addWidget(data_selection)
        vlayout.addWidget(curr_slice)
        vlayout.addWidget(whole_dataset)
        vlayout.addWidget(denoise_type)
        vlayout.addWidget(tv_L1)
        vlayout.addWidget(tv_L2)
        vlayout.addWidget(tv_ROF)
        vlayout.addWidget(btns)

        hlayout.addLayout(vlayout)
        hlayout.addLayout(form)
        
        dialog.setLayout(hlayout)
        #dialog.setStyleSheet('color: white')
        dialog.setStyleSheet("QLineEdit"
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
        dialog.exec_()

    def signal_ok(self):
        self.alpha = float(self.qline_alpha.text())
        self.lambd = float(self.qline_lambd.text())
        self.iter = int(self.qline_iter.text())  

        print(self.tv_group.checkedId())
        
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


    #def signal_cancel(self, dialog_object):
        
        #dialog_object.close()



# class DialogDenoise(QDialog):

#   def __init__(self, parent = None):
#       super(DialogDenoise, self).__init__(parent)

#       create Widgets, order them in Layout, connect signal to slots

#       initialize all widget with self.

#   def MethodsForSignals...


class SidebarHeading(QLabel):  # pylint: disable=R0903
    """Widget for sidebar separators and headings."""

    def __init__(self, text="", first=False):
        QLabel.__init__(self)
        self.setText(text)
        role = "widget_heading__first" if first else "widget_heading"
        self.setProperty("role", role)
        self.setMinimumHeight(24)
        self.setMaximumHeight(24)
