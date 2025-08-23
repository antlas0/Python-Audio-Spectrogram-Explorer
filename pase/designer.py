import numpy as np
import pandas as pd
import datetime as dt
from PyQt6 import QtCore, QtGui, QtWidgets
from matplotlib.backends.backend_qt5agg import (
    FigureCanvasQTAgg,
    NavigationToolbar2QT as NavigationToolbar,
)
from matplotlib.figure import Figure


class MplCanvas(FigureCanvasQTAgg):
    # def __init__(self, parent=None, width=5, height=4, dpi=100):
    #     self.fig = Figure(figsize=(width, height), dpi=dpi)
    #     self.axes = self.fig.add_subplot(111)
    #     super(MplCanvas, self).__init__(self.fig)

    def __init__(self, parent=None, dpi=150):
        self.fig = Figure(figsize=None, dpi=dpi)
        # self.axes = self.fig.add_subplot(111)
        # self.axes.set_facecolor('gray')

        super(MplCanvas, self).__init__(self.fig)


def populate_ui(parent):
    # parent.call_time=pd.Series()
    # parent.call_frec=pd.Series()

    parent.f_min = QtWidgets.QLineEdit(parent)
    parent.f_min.setText("10")
    parent.f_max = QtWidgets.QLineEdit(parent)
    parent.f_max.setText("16000")
    parent.t_length = QtWidgets.QLineEdit(parent)
    parent.t_length.setText("120")
    parent.db_saturation = QtWidgets.QLineEdit(parent)
    parent.db_saturation.setText("155")
    parent.db_vmin = QtWidgets.QLineEdit(parent)
    parent.db_vmin.setText("30")
    parent.db_vmax = QtWidgets.QLineEdit(parent)
    parent.db_vmax.setText("")
    # parent.fft_size = QtWidgets.QLineEdit(parent)
    # parent.fft_size.setText('32768')
    parent.fft_size = QtWidgets.QComboBox(parent)
    parent.fft_size.addItem("1024")
    parent.fft_size.addItem("2048")
    parent.fft_size.addItem("4096")
    parent.fft_size.addItem("8192")
    parent.fft_size.addItem("16384")
    parent.fft_size.addItem("32768")
    parent.fft_size.addItem("65536")
    parent.fft_size.addItem("131072")
    parent.fft_size.setCurrentIndex(4)
    parent.checkbox_background = QtWidgets.QCheckBox("Remove background")
    parent.checkbox_background.setChecked(False)


    parent.colormap_plot = QtWidgets.QComboBox(parent)
    parent.colormap_plot.addItem("plasma")
    parent.colormap_plot.addItem("viridis")
    parent.colormap_plot.addItem("inferno")
    parent.colormap_plot.addItem("gist_gray")
    parent.colormap_plot.addItem("gist_yarg")
    parent.colormap_plot.setCurrentIndex(2)

    # parent.fft_overlap = QtWidgets.QLineEdit(parent)
    # parent.fft_overlap.setText('0.9')

    parent.fft_overlap = QtWidgets.QComboBox(parent)
    parent.fft_overlap.addItem("0.2")
    parent.fft_overlap.addItem("0.5")
    parent.fft_overlap.addItem("0.7")
    parent.fft_overlap.addItem("0.9")
    parent.fft_overlap.setCurrentIndex(3)

    parent.filename_timekey = QtWidgets.QLineEdit(parent)
    parent.filename_timekey.textChanged.connect(parent.timekey_save_to_settings)
    # parent.filename_timekey.setText('aural_%Y_%m_%d_%H_%M_%S.wav')

    parent.playbackspeed = QtWidgets.QComboBox(parent)
    parent.playbackspeed.addItem("0.5")
    parent.playbackspeed.addItem("1")
    parent.playbackspeed.addItem("2")
    parent.playbackspeed.addItem("5")
    parent.playbackspeed.addItem("10")
    parent.playbackspeed.setCurrentIndex(1)

    parent.time = dt.datetime(2000, 1, 1, 0, 0, 0)
    parent.f = None
    parent.t = [-1, -1]
    parent.Sxx = None
    parent.draw_x = pd.Series(dtype="float")
    parent.draw_y = pd.Series(dtype="float")

    parent.plotwindow_startsecond = float(parent.t_length.text())
    # parent.plotwindow_length=120
    parent.filecounter = -1
    parent.filenames = np.array([])
    parent.current_audiopath = None

    parent.detectiondf = pd.DataFrame([])

    parent.fft_size.currentIndexChanged.connect(parent.new_fft_size_selected)
    parent.colormap_plot.currentIndexChanged.connect(parent.plot_spectrogram)
    parent.checkbox_background.stateChanged.connect(parent.plot_spectrogram)
    parent.checkbox_logscale.stateChanged.connect(parent.plot_spectrogram)

    # parent.checkbox_log = QtWidgets.QCheckBox("Real-time Logging")
    # parent.checkbox_log.toggled.connect(parent.func_logging)

    button_export_all_spectrograms = QtWidgets.QPushButton("Plot all spectrograms")
    button_export_all_spectrograms.clicked.connect(parent.export_all_spectrograms)

    parent.annotate_button = QtWidgets.QPushButton("🖊️ Annotate")
    parent.annotate_button.setToolTip("Write annotation work and select it with annotation checkbox. Then drag left click on a zone to annotate. Escape to discard, Enter to save to a file.")
    parent.annotate_button.setCheckable(False)
    parent.annotate_button.clicked.connect(parent.func_annotate)

    parent.load_annotations_button = QtWidgets.QPushButton("📁 Load annotations")
    parent.load_annotations_button.setToolTip("Load annotations CSV files. To add new annotations, click on Annotate. New annotations will be appended into loaded file of corresponding spectrogram.")
    parent.load_annotations_button.setCheckable(False)
    parent.load_annotations_button.clicked.connect(parent.load_annotations)

    parent.draw_button = QtWidgets.QPushButton("🖌️ Draw")
    parent.draw_button.setToolTip("Draw a polygon. Double click to add a point, Escape to discard, Enter to save into a file.")
    parent.draw_button.setCheckable(False)
    parent.draw_button.clicked.connect(parent.func_draw_shape)

    parent.load_drawing_button = QtWidgets.QPushButton("📁 Load drawing")
    parent.load_drawing_button.setToolTip("Load drawing file. To add new drawings, click on Draw. New drawings will be added into loaded file of corresponding spectrogram.")
    parent.load_drawing_button.setCheckable(False)
    parent.load_drawing_button.clicked.connect(parent.load_drawing)

    parent.disp_all_button = QtWidgets.QPushButton("Display all")
    parent.disp_all_button.setToolTip("Display both annotations and drawing created on a spectrogram, if exist.")
    parent.disp_all_button.setCheckable(False)
    parent.disp_all_button.clicked.connect(parent.display_all_artifacts)

    parent.clear_spectro_button = QtWidgets.QPushButton("🗑️ Clear")
    parent.clear_spectro_button.setToolTip("Clear annotations and drawing from a spectrogram.")
    parent.clear_spectro_button.setCheckable(False)
    parent.clear_spectro_button.clicked.connect(parent.clear_spectrogram)


    ####### play audio
    button_play_audio = QtWidgets.QPushButton("⏯️")
    button_play_audio.clicked.connect(parent.func_playaudio)

    button_save_audio = QtWidgets.QPushButton("Export selected audio")
    button_save_audio.clicked.connect(parent.func_saveaudio)

    button_save_video = QtWidgets.QPushButton("Export video")

    button_save_video.clicked.connect(parent.func_save_video)

    ############# menu
    menuBar = parent.menuBar()

    # Creating menus using a title
    exportMenu = menuBar.addMenu("Export")
    e1 = exportMenu.addAction("Spectrogram as .wav file")
    e1.triggered.connect(parent.func_saveaudio)
    e2 = exportMenu.addAction("Spectrogram as animated video")
    e2.triggered.connect(parent.func_save_video)
    e3 = exportMenu.addAction("Spectrogram as .csv table")
    e3.triggered.connect(parent.export_zoomed_sgram_as_csv)
    e4 = exportMenu.addAction("All files as spectrogram images")
    e4.triggered.connect(parent.export_all_spectrograms)
    e6 = exportMenu.addAction("Automatic detections as .csv table")
    e6.triggered.connect(parent.export_automatic_detector)

    autoMenu = menuBar.addMenu("Automatic detection")
    a1 = autoMenu.addAction("Shapematching on current file")
    a1.triggered.connect(parent.automatic_detector_shapematching)
    # a3 = autoMenu.addAction("Shapematching on all files")
    # a3.triggered.connect(parent.automatic_detector_shapematching_allfiles)

    a2 = autoMenu.addAction("Spectrogram correlation on current file")
    a2.triggered.connect(parent.automatic_detector_specgram_corr)

    # a4 = autoMenu.addAction("Spectrogram correlation on all files")
    # a4.triggered.connect(parent.automatic_detector_specgram_corr_allfiles)

    a5 = autoMenu.addAction("Show regions based on threshold")
    a5.triggered.connect(parent.plot_spectrogram_threshold)

    aboutMenu = menuBar.addAction("About")
    aboutMenu.triggered.connect(parent.aboutfunc)

    quitMenu = menuBar.addAction("Quit")
    quitMenu.triggered.connect(parent.exitfunc)


    #################

    ######## layout
    parent.outer_layout = QtWidgets.QVBoxLayout()

    parent.top2_layout = QtWidgets.QHBoxLayout()

    # parent.top2_layout.addWidget(parent.checkbox_log)

    parent.top2_layout.addWidget(QtWidgets.QLabel("Timestamp:"))
    parent.top2_layout.addWidget(parent.filename_timekey)
    parent.top2_layout.addWidget(QtWidgets.QLabel("f_min[Hz]:"))
    parent.top2_layout.addWidget(parent.f_min)
    parent.top2_layout.addWidget(QtWidgets.QLabel("f_max[Hz]:"))
    parent.top2_layout.addWidget(parent.f_max)
    parent.top2_layout.addWidget(QtWidgets.QLabel("Spec. length [sec]:"))
    parent.top2_layout.addWidget(parent.t_length)

    parent.top2_layout.addWidget(QtWidgets.QLabel("Saturation dB:"))
    parent.top2_layout.addWidget(parent.db_saturation)

    parent.top2_layout.addWidget(QtWidgets.QLabel("dB min:"))
    parent.top2_layout.addWidget(parent.db_vmin)
    parent.top2_layout.addWidget(QtWidgets.QLabel("dB max:"))
    parent.top2_layout.addWidget(parent.db_vmax)

    # annotation label area
    parent.top3_layout = QtWidgets.QHBoxLayout()
    parent.top3_layout.addWidget(QtWidgets.QLabel("Annotation labels:"))

    parent.checkbox_an_1 = QtWidgets.QCheckBox()
    parent.top3_layout.addWidget(parent.checkbox_an_1)
    parent.an_1 = QtWidgets.QLineEdit(parent)
    parent.top3_layout.addWidget(parent.an_1)
    parent.an_1.setText("")

    parent.checkbox_an_2 = QtWidgets.QCheckBox()
    parent.top3_layout.addWidget(parent.checkbox_an_2)
    parent.an_2 = QtWidgets.QLineEdit(parent)
    parent.top3_layout.addWidget(parent.an_2)
    parent.an_2.setText("")

    parent.checkbox_an_3 = QtWidgets.QCheckBox()
    parent.top3_layout.addWidget(parent.checkbox_an_3)
    parent.an_3 = QtWidgets.QLineEdit(parent)
    parent.top3_layout.addWidget(parent.an_3)
    parent.an_3.setText("")

    parent.checkbox_an_4 = QtWidgets.QCheckBox()
    parent.top3_layout.addWidget(parent.checkbox_an_4)
    parent.an_4 = QtWidgets.QLineEdit(parent)
    parent.top3_layout.addWidget(parent.an_4)
    parent.an_4.setText("")

    parent.checkbox_an_5 = QtWidgets.QCheckBox()
    parent.top3_layout.addWidget(parent.checkbox_an_5)
    parent.an_5 = QtWidgets.QLineEdit(parent)
    parent.top3_layout.addWidget(parent.an_5)
    parent.an_5.setText("")

    parent.checkbox_an_6 = QtWidgets.QCheckBox()
    parent.top3_layout.addWidget(parent.checkbox_an_6)
    parent.an_6 = QtWidgets.QLineEdit(parent)
    parent.top3_layout.addWidget(parent.an_6)
    parent.an_6.setText("")

    parent.bg = QtWidgets.QButtonGroup()
    parent.bg.addButton(parent.checkbox_an_1, 1)
    parent.bg.addButton(parent.checkbox_an_2, 2)
    parent.bg.addButton(parent.checkbox_an_3, 3)
    parent.bg.addButton(parent.checkbox_an_4, 4)
    parent.bg.addButton(parent.checkbox_an_5, 5)
    parent.bg.addButton(parent.checkbox_an_6, 6)

    # combine layouts together

    parent.bodylayout = QtWidgets.QHBoxLayout()

    parent.sidepanel_layout = QtWidgets.QVBoxLayout()
    open_library_btn = QtWidgets.QPushButton("📁 Open library")
    open_library_btn.clicked.connect(parent.set_user_library)
    open_output_dir_btn = QtWidgets.QPushButton("📁 Open output directory")
    open_output_dir_btn.clicked.connect(parent.set_output_directory)

    parent.filesview.setMaximumWidth(500)
    parent.outputview.setMaximumWidth(500)
    parent.sidepanel_layout.addWidget(open_library_btn)
    parent.sidepanel_layout.addWidget(parent.filesview)
    parent.sidepanel_layout.addWidget(open_output_dir_btn)
    parent.sidepanel_layout.addWidget(parent.outputview)
    parent.plot_layout = QtWidgets.QVBoxLayout()
    tnav = NavigationToolbar(parent.canvas, parent)

    parent.toolbar = QtWidgets.QToolBar()

    parent.move_layout = QtWidgets.QHBoxLayout()
    parent.move_layout.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
    parent.button_plot_prev_spectro = QtWidgets.QPushButton("⬅️")
    parent.button_plot_prev_spectro.clicked.connect(parent.plot_previous_spectro)
    parent.move_layout.addWidget(parent.button_plot_prev_spectro)

    ss = "  "
    parent.move_layout.addWidget(QtWidgets.QLabel(ss))

    parent.button_plot_next_spectro = QtWidgets.QPushButton("➡️")
    parent.button_plot_next_spectro.clicked.connect(parent.plot_next_spectro)
    parent.move_layout.addWidget(parent.button_plot_next_spectro)

    parent.move_layout.addWidget(QtWidgets.QLabel(ss))

    parent.move_layout.addWidget(button_play_audio)
    parent.move_layout.addWidget(QtWidgets.QLabel(ss))

    parent.move_layout.addWidget(QtWidgets.QLabel("Playback speed:"))
    parent.move_layout.addWidget(QtWidgets.QLabel(ss))
    parent.move_layout.addWidget(parent.playbackspeed)
    parent.move_layout.addWidget(QtWidgets.QLabel(ss))

    parent.toolbar.addWidget(QtWidgets.QLabel(ss))

    action_layout = QtWidgets.QHBoxLayout()
    action_layout.addWidget(parent.annotate_button)
    action_layout.addWidget(parent.load_annotations_button)
    action_layout.addWidget(parent.draw_button)
    action_layout.addWidget(parent.load_drawing_button)
    action_layout.addWidget(parent.disp_all_button)
    action_layout.addWidget(parent.clear_spectro_button)
    parent.toolbar.addSeparator()
    parent.toolbar.addWidget(parent.checkbox_logscale)
    parent.toolbar.addWidget(parent.checkbox_background)
    parent.toolbar.addSeparator()
    parent.toolbar.addWidget(QtWidgets.QLabel("fft_size[bits]:"))
    parent.toolbar.addWidget(QtWidgets.QLabel(ss))
    parent.toolbar.addWidget(parent.fft_size)
    parent.toolbar.addWidget(QtWidgets.QLabel(ss))
    parent.toolbar.addWidget(QtWidgets.QLabel("fft_overlap[0-1]:"))
    parent.toolbar.addWidget(QtWidgets.QLabel(ss))
    parent.toolbar.addWidget(parent.fft_overlap)

    parent.toolbar.addWidget(QtWidgets.QLabel(ss))

    parent.toolbar.addWidget(QtWidgets.QLabel("Colormap:"))
    parent.toolbar.addWidget(QtWidgets.QLabel(ss))
    parent.toolbar.addWidget(parent.colormap_plot)
    parent.toolbar.addWidget(QtWidgets.QLabel(ss))

    parent.toolbar.addSeparator()

    parent.plot_layout.addWidget(parent.toolbar)
    parent.plot_layout.addLayout(action_layout)
    parent.plot_layout.addWidget(tnav)
    parent.plot_layout.addWidget(parent.canvas)
    parent.plot_layout.addLayout(parent.move_layout)

    # parent.outer_layout.addLayout(top_layout)
    parent.outer_layout.addLayout(parent.top2_layout)
    parent.outer_layout.addLayout(parent.top3_layout)

    parent.bodylayout.addLayout(parent.sidepanel_layout)
    parent.bodylayout.addLayout(parent.plot_layout)

    parent.notification_bar = QtWidgets.QLabel()
    footerlayout = QtWidgets.QVBoxLayout()
    footerlayout.addWidget(parent.notification_bar)

    parent.outer_layout.addLayout(parent.bodylayout)
    parent.outer_layout.addLayout(footerlayout)

    # parent.setLayout(parent.outer_layout)

    # Create a placeholder widget to hold our parent.toolbar and canvas.
    widget = QtWidgets.QWidget()
    widget.setLayout(parent.outer_layout)
    parent.setCentralWidget(widget)

    #### hotkeys
    parent.msgSc1 = QtGui.QShortcut(QtCore.Qt.Key.Key_Right, parent)
    parent.msgSc1.activated.connect(parent.plot_next_spectro)
    parent.msgSc2 = QtGui.QShortcut(QtCore.Qt.Key.Key_Left, parent)
    parent.msgSc2.activated.connect(parent.plot_previous_spectro)
    parent.msgSc3 = QtGui.QShortcut(QtCore.Qt.Key.Key_Space, parent)
    parent.msgSc3.activated.connect(parent.func_playaudio)

    parent.show()
