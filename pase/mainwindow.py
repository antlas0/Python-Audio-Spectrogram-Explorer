import os
import time
import numpy as np
import pandas as pd
import datetime as dt

from PyQt6 import QtCore, QtWidgets
from PyQt6.QtCore import QDir, QModelIndex
from PyQt6.QtGui import QFileSystemModel, QShortcut

import soundfile as sf
import simpleaudio as sa
from moviepy import VideoClip, AudioFileClip

from matplotlib.backends.backend_qt5agg import (
    FigureCanvasQTAgg,
    NavigationToolbar2QT as NavigationToolbar,
)
from matplotlib.widgets import RectangleSelector
from matplotlib.path import Path
from matplotlib import pyplot as plt

import scipy.io.wavfile as wav
from scipy import signal

from skimage import data, filters, measure, morphology
from skimage.morphology import (
    erosion,
    dilation,
    opening,
    closing,  # noqa
    white_tophat,
)
from skimage.morphology import disk  # noqa
from skimage.transform import rescale, resize, downscale_local_mean
from scipy.signal import find_peaks
from skimage.feature import match_template
from typing import List

from .designer import populate_ui, MplCanvas
from .resources import str2bool, InputData, AudioSample, FFTSample, MAX_LENGHT_SEC, LICENCE_STR



# Copied from MoviePy 1, because it has been removed in version 2.
# https://github.com/Zulko/moviepy/blob/db19920764b5cb1d8aa6863019544fd8ae0d3cce/moviepy/video/io/bindings.py#L18C1-L32C32
# With some updates to Matplotlib 3.8.
def mplfig_to_npimage(fig):
    """Converts a matplotlib figure to a RGB frame after updating the canvas"""
    from matplotlib.backends.backend_agg import FigureCanvasAgg

    canvas = FigureCanvasAgg(fig)
    canvas.draw()  # update/draw the elements

    # get the width and the height to resize the matrix
    l, b, w, h = canvas.figure.bbox.bounds
    w, h = int(w), int(h)

    #  exports the canvas to a memory view and then to a numpy nd.array
    mem_view = canvas.buffer_rgba()  # Update to Matplotlib 3.8
    image = np.asarray(mem_view)
    return image[:, :, :3]  # Return only RGB, not alpha.


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, *args, **kwargs):
        super(MainWindow, self).__init__(*args, **kwargs)
        self.settings = QtCore.QSettings("antlas0", "PASE")
        self.filesview = QtWidgets.QTreeView()
        self.filesview.doubleClicked.connect(self.on_file_double_clicked)
        self.outputview = QtWidgets.QTreeView()
        self.outputview.doubleClicked.connect(self.on_file_double_clicked)
        self.canvas = MplCanvas(self, dpi=150)
        self._filepointer = -1
        self._input_data: List[InputData] = []
        self._detectiondf = None
        self.library_directory = self.settings.value(
            "library_directory", os.path.abspath(os.getcwd())
        )
        self._output_directory = self.settings.value(
            "output_directory", os.path.abspath(os.getcwd())
        )
        self.setup_ui()
        self.setup()
        self.filename_timekey.setText(self.settings.value("timekey", ""))

    def setup_ui(self):
        self.checkbox_logscale = QtWidgets.QCheckBox("Log. scale")
        self.checkbox_logscale.stateChanged.connect(self.update_settings)
        self.checkbox_logscale.setChecked(
            str2bool(self.settings.value("log_scale", "false"))
        )
        populate_ui(self)

    def setup(self):
        if (directory := self.settings.value("library_directory", None)) is not None:
            self.open_library(directory)
        if (directory := self.settings.value("output_directory", None)) is not None:
            self.open_output_directory(directory)

    def aboutfunc(self):
        dlg = QtWidgets.QMessageBox(self)
        dlg.setWindowTitle("About")
        dlg.setText("Work by <a href=\"https://github.com/antlas0\">antlas0</a>,<br/>built on top of original development by <a href=\"https://github.com/sebastianmenze/Python-Audio-Spectrogram-Explorer\">sebastianmenze</a><br/><br/>Licence:<br/>" + LICENCE_STR)
        dlg.exec()

    def exitfunc(self):
        QtWidgets.QApplication.instance().quit
        self.close()

    def update_settings(self):
        self.settings.setValue("log_scale", str(self.checkbox_logscale.isChecked()))

    def notify_message(self, message:str):
        self.notification_bar.setText(message)
        self.repaint()

    def timekey_save_to_settings(self, text:str):
        self.settings.setValue("timekey", text)

    ########################
    def find_regions(self, db_threshold):
        y1 = int(self.f_min.text())
        y2 = int(self.f_max.text())
        if y2 > (self._input_data[self._filepointer].audio_data.fs / 2):
            y2 = self._input_data[self._filepointer].audio_data.fs / 2
        t1 = self.plotwindow_startsecond
        t2 = self.plotwindow_startsecond + self.plotwindow_length

        ix_time = np.where((self._input_data[self._filepointer].fft_data.t>= t1) & (self._input_data[self._filepointer].fft_data.t< t2))[0]
        ix_f = np.where((self._input_data[self._filepointer].fft_data.f >= y1) & (self._input_data[self._filepointer].fft_data.f < y2))[0]

        # f_lim=[self.f_min,self.f_max]
        t = self._input_data[self._filepointer].fft_data.t[ix_time]

        # db_threshold=10
        minimum_patcharea = 5

        plotsxx = self._input_data[self._filepointer].fft_data.ssx[
            int(ix_f[0]) : int(ix_f[-1]), int(ix_time[0]) : int(ix_time[-1])
        ]
        spectrog = 10 * np.log10(plotsxx)

        # filter out background
        spec_mean = np.median(spectrog, axis=1)
        sxx_background = np.transpose(
            np.broadcast_to(spec_mean, np.transpose(spectrog).shape)
        )
        z = spectrog - sxx_background

        # z=spectrog - np.min(spectrog.flatten())

        # rectime= pd.to_timedelta( t ,'s')
        # spg=pd.DataFrame(np.transpose(spectrog),index=rectime)
        # bg=spg.resample('3min').mean().copy()
        # bg=bg.resample('1s').interpolate(method='time')
        # bg=    bg.reindex(rectime,method='nearest')

        # background=np.transpose(bg.values)
        # z=spectrog-background

        # Binary image, post-process the binary mask and compute labels
        mask = z > db_threshold
        mask = morphology.remove_small_objects(mask, 50, connectivity=30)
        mask = morphology.remove_small_holes(mask, 50, connectivity=30)

        mask = closing(mask, disk(3))
        # op_and_clo = opening(closed,  disk(1) )

        labels = measure.label(mask)

        probs = measure.regionprops_table(
            labels,
            spectrog,
            properties=[
                "label",
                "area",
                "mean_intensity",
                "orientation",
                "major_axis_length",
                "minor_axis_length",
                "weighted_centroid",
                "bbox",
            ],
        )
        df = pd.DataFrame(probs)

        # get corect f anf t
        ff = self._input_data[self._filepointer].fft_data.f[ix_f[0] : ix_f[-1]]
        ix = df["bbox-0"] > len(ff) - 1
        df.loc[ix, "bbox-0"] = len(ff) - 1
        ix = df["bbox-2"] > len(ff) - 1
        df.loc[ix, "bbox-2"] = len(ff) - 1

        df["f-1"] = ff[df["bbox-0"]]
        df["f-2"] = ff[df["bbox-2"]]
        df["f-width"] = df["f-2"] - df["f-1"]

        ix = df["bbox-1"] > len(t) - 1
        df.loc[ix, "bbox-1"] = len(t) - 1
        ix = df["bbox-3"] > len(t) - 1
        df.loc[ix, "bbox-3"] = len(t) - 1

        df["t-1"] = t[df["bbox-1"]]
        df["t-2"] = t[df["bbox-3"]]
        df["duration"] = df["t-2"] - df["t-1"]

        indices = np.where(
            (df["area"] < minimum_patcharea)
            | (df["bbox-3"] - df["bbox-1"] < 3)
            | (df["bbox-2"] - df["bbox-0"] < 3)
        )[0]
        df = df.drop(indices)
        df = df.reset_index()

        df["id"] = np.arange(len(df))

        # df['filename']=audiopath

        # get region dict
        # sgram={}
        patches = {}
        p_t_dict = {}
        p_f_dict = {}

        for ix in range(len(df)):
            m = labels == df.loc[ix, "label"]
            ix1 = df.loc[ix, "bbox-1"]
            ix2 = df.loc[ix, "bbox-3"]
            jx1 = df.loc[ix, "bbox-0"]
            jx2 = df.loc[ix, "bbox-2"]

            patch = m[jx1:jx2, ix1:ix2]
            pt = t[ix1:ix2]
            pt = pt - pt[0]
            pf = ff[jx1:jx2]

            # contour = measure.find_contours(m, 0.5)[0]
            # y, x = contour.T

            patches[df["id"][ix]] = patch
            p_t_dict[df["id"][ix]] = pt
            p_f_dict[df["id"][ix]] = pf

            # ix1=ix1-10
            # if ix1<=0: ix1=0
            # ix2=ix2+10
            # if ix2>=spectrog.shape[1]: ix2=spectrog.shape[1]-1
            # sgram[ df['id'][ix]  ] = spectrog[:,ix1:ix2]
        self._detectiondf = df
        self.patches = patches
        self.p_t_dict = p_t_dict
        self.p_f_dict = p_f_dict

        self.region_labels = labels

        # return df, patches,p_t_dict,p_f_dict

    def match_bbox_and_iou(self, template):
        shape_f = template["Frequency_in_Hz"].values
        shape_t = template["Time_in_s"].values
        shape_t = shape_t - shape_t.min()

        df = self._detectiondf
        patches = self.patches
        p_t_dict = self.p_t_dict
        p_f_dict = self.p_f_dict

        # f_lim=[ shape_f.min()-10 ,shape_f.max()+10  ]

        # score_smc=[]
        score_ioubox = []
        smc_rs = []

        for ix in df.index:
            # breakpoint()
            patch = patches[ix]
            pf = p_f_dict[ix]
            pt = p_t_dict[ix]
            pt = pt - pt[0]

            if df.loc[ix, "f-1"] < shape_f.min():
                f1 = df.loc[ix, "f-1"]
            else:
                f1 = shape_f.min()
            if df.loc[ix, "f-2"] > shape_f.max():
                f2 = df.loc[ix, "f-2"]
            else:
                f2 = shape_f.max()

            # f_lim=[ f1,f2  ]

            time_step = np.diff(pt)[0]
            f_step = np.diff(pf)[0]
            k_f = np.arange(f1, f2, f_step)

            if pt.max() > shape_t.max():
                k_t = pt

            else:
                k_t = np.arange(0, shape_t.max(), time_step)

            ### iou bounding box

            iou_kernel = np.zeros([k_f.shape[0], k_t.shape[0]])
            ixp2 = np.where((k_t >= shape_t.min()) & (k_t <= shape_t.max()))[0]
            ixp1 = np.where((k_f >= shape_f.min()) & (k_f <= shape_f.max()))[0]
            iou_kernel[ixp1[0] : ixp1[-1], ixp2[0] : ixp2[-1]] = 1

            iou_patch = np.zeros([k_f.shape[0], k_t.shape[0]])
            ixp2 = np.where((k_t >= pt[0]) & (k_t <= pt[-1]))[0]
            ixp1 = np.where((k_f >= pf[0]) & (k_f <= pf[-1]))[0]
            iou_patch[ixp1[0] : ixp1[-1], ixp2[0] : ixp2[-1]] = 1

            intersection = iou_kernel.astype("bool") & iou_patch.astype("bool")
            union = iou_kernel.astype("bool") | iou_patch.astype("bool")
            iou_bbox = np.sum(intersection) / np.sum(union)
            score_ioubox.append(iou_bbox)

            patch_rs = resize(patch, (50, 50))
            n_resize = 50
            k_t = np.linspace(0, shape_t.max(), n_resize)
            k_f = np.linspace(shape_f.min(), shape_f.max(), n_resize)
            kk_t, kk_f = np.meshgrid(k_t, k_f)
            # kernel=np.zeros( [ k_f.shape[0] ,k_t.shape[0] ] )
            x, y = kk_t.flatten(), kk_f.flatten()
            points = np.vstack((x, y)).T
            p = Path(list(zip(shape_t, shape_f)))  # make a polygon
            grid = p.contains_points(points)
            kernel_rs = grid.reshape(
                kk_t.shape
            )  # now you have a mask with points inside a polygon
            smc_rs.append(
                np.sum(kernel_rs.astype("bool") == patch_rs.astype("bool"))
                / len(patch_rs.flatten())
            )

        smc_rs = np.array(smc_rs)
        score_ioubox = np.array(score_ioubox)

        # df['score'] =score_ioubox * (smc_rs-.5)/.5
        score = score_ioubox * (smc_rs - 0.5) / 0.5
        return score
        # self._detectiondf = df.copy()

    def automatic_detector_specgram_corr(self):
        # open template
        self._detectiondf = pd.DataFrame([])

        templatefiles, ok1 = QtWidgets.QFileDialog.getOpenFileNames(
            self, "QFileDialog.getOpenFileNames()", self._output_directory, "CSV file (*.csv)"
        )
        if ok1:
            templates = []
            for fnam in templatefiles:
                template = pd.read_csv(fnam, index_col=0)
                templates.append(template)

            corrscore_threshold, ok = QtWidgets.QInputDialog.getDouble(
                self,
                "Input Dialog",
                "Enter correlation threshold in (0-1):",
                decimals=2,
            )
            if corrscore_threshold > 1:
                corrscore_threshold = 1
            if corrscore_threshold < 0:
                corrscore_threshold = 0

            # print(templates)
            # print(templates[0])

            if templates[0].columns[0] == "Time_in_s":
                # print(template)
                offset_f = 10
                offset_t = 0.5

                # shape_f=template['Frequency_in_Hz'].values
                # shape_t=template['Time_in_s'].values
                # shape_t=shape_t-shape_t.min()
                shape_f = np.array([])
                shape_t_raw = np.array([])
                for template in templates:
                    shape_f = np.concatenate(
                        [shape_f, template["Frequency_in_Hz"].values]
                    )
                    shape_t_raw = np.concatenate(
                        [shape_t_raw, template["Time_in_s"].values]
                    )
                shape_t = shape_t_raw - shape_t_raw.min()

                f_lim = [shape_f.min() - offset_f, shape_f.max() + offset_f]
                k_length_seconds = shape_t.max() + offset_t * 2

                # generate kernel
                time_step = np.diff(self._input_data[self._filepointer].fft_data.t)[0]

                k_t = np.linspace(
                    0, k_length_seconds, int(k_length_seconds / time_step)
                )
                ix_f = np.where((self._input_data[self._filepointer].fft_data.f >= f_lim[0]) & (self._input_data[self._filepointer].fft_data.f <= f_lim[1]))[0]
                k_f = self._input_data[self._filepointer].fft_data.f[ix_f[0] : ix_f[-1]]
                # k_f=np.linspace(f_lim[0],f_lim[1], int( (f_lim[1]-f_lim[0]) /f_step)  )

                kk_t, kk_f = np.meshgrid(k_t, k_f)
                kernel_background_db = 0
                kernel_signal_db = 1
                kernel = np.ones([k_f.shape[0], k_t.shape[0]]) * kernel_background_db
                # find wich grid points are inside the shape
                x, y = kk_t.flatten(), kk_f.flatten()
                points = np.vstack((x, y)).T
                # p = Path(list(zip(shape_t, shape_f))) # make a polygon
                # grid = p.contains_points(points)
                # mask = grid.reshape(kk_t.shape) # now you have a mask with points inside a polygon
                # kernel[mask]=kernel_signal_db
                for template in templates:
                    shf = template["Frequency_in_Hz"].values
                    st = template["Time_in_s"].values
                    st = st - shape_t_raw.min()
                    p = Path(list(zip(st, shf)))  # make a polygon
                    grid = p.contains_points(points)
                    kern = grid.reshape(
                        kk_t.shape
                    )  # now you have a mask with points inside a polygon
                    kernel[kern > 0] = kernel_signal_db

                # print(kernel)

                ix_f = np.where((self._input_data[self._filepointer].fft_data.f >= f_lim[0]) & (self._input_data[self._filepointer].fft_data.f <= f_lim[1]))[0]
                spectrog = 10 * np.log10(self._input_data[self._filepointer].fft_data.ssx[ix_f[0] : ix_f[-1], :])

                result = match_template(spectrog, kernel)
                corr_score = result[0, :]
                t_score = np.linspace(
                    self._input_data[self._filepointer].fft_data.t[int(kernel.shape[1] / 2)],
                    self._input_data[self._filepointer].fft_data.t[-int(kernel.shape[1] / 2)],
                    corr_score.shape[0],
                )

                peaks_indices = find_peaks(corr_score, height=corrscore_threshold)[0]

                t1 = []
                t2 = []
                f1 = []
                f2 = []
                score = []

                if len(peaks_indices) > 0:
                    t2_old = 0
                    for ixpeak in peaks_indices:
                        tstar = t_score[ixpeak] - k_length_seconds / 2 - offset_t
                        tend = t_score[ixpeak] + k_length_seconds / 2 - offset_t
                        # if tstar>t2_old:
                        t1.append(tstar)
                        t2.append(tend)
                        f1.append(f_lim[0] + offset_f)
                        f2.append(f_lim[1] - offset_f)
                        score.append(corr_score[ixpeak])
                        # t2_old=tend
                    df = pd.DataFrame()
                    df["t-1"] = t1
                    df["t-2"] = t2
                    df["f-1"] = f1
                    df["f-2"] = f2
                    df["score"] = score

                    self._detectiondf = df.copy()
                    self._detectiondf["audiofilename"] = self._input_data[self._filepointer].filename
                    self._detectiondf["threshold"] = corrscore_threshold

                    self.plot_spectrogram()
            else:  # image kernel
                template = templates[0]

                k_length_seconds = float(template.columns[-1]) - float(
                    template.columns[0]
                )

                f_lim = [int(template.index[0]), int(template.index[-1])]
                ix_f = np.where((self._input_data[self._filepointer].fft_data.f >= f_lim[0]) & (self._input_data[self._filepointer].fft_data.f <= f_lim[1]))[0]
                spectrog = 10 * np.log10(self._input_data[self._filepointer].fft_data.ssx[ix_f[0] : ix_f[-1], :])
                specgram_t_step = self._input_data[self._filepointer].fft_data.t[1] - self._input_data[self._filepointer].fft_data.t[0]
                n_f = spectrog.shape[0]
                n_t = int(k_length_seconds / specgram_t_step)

                kernel = resize(template.values, [n_f, n_t])

                result = match_template(spectrog, kernel)
                corr_score = result[0, :]
                t_score = np.linspace(
                    self._input_data[self._filepointer].fft_data.t[int(kernel.shape[1] / 2)],
                    self._input_data[self._filepointer].fft_data.t[-int(kernel.shape[1] / 2)],
                    corr_score.shape[0],
                )

                peaks_indices = find_peaks(corr_score, height=corrscore_threshold)[0]

                # print(corr_score)

                t1 = []
                t2 = []
                f1 = []
                f2 = []
                score = []

                if len(peaks_indices) > 0:
                    t2_old = 0
                    for ixpeak in peaks_indices:
                        tstar = t_score[ixpeak] - k_length_seconds / 2
                        tend = t_score[ixpeak] + k_length_seconds / 2
                        # if tstar>t2_old:
                        t1.append(tstar)
                        t2.append(tend)
                        f1.append(f_lim[0])
                        f2.append(f_lim[1])
                        score.append(corr_score[ixpeak])
                        t2_old = tend
                    df = pd.DataFrame()
                    df["t-1"] = t1
                    df["t-2"] = t2
                    df["f-1"] = f1
                    df["f-2"] = f2
                    df["score"] = score

                    self._detectiondf = df.copy()
                    self._detectiondf["audiofilename"] = self._input_data[self._filepointer].filename
                    self._detectiondf["threshold"] = corrscore_threshold

            print(self._detectiondf)

            self.plot_spectrogram()

    # def automatic_detector_specgram_corr_allfiles(self):
    #     text = (
    #         "Are you sure you want to run the detector over "
    #         + str(len(self._input_data))
    #         + " ?"
    #     )

    #     msg = QtWidgets.QMessageBox.information(
    #         self,
    #         "Detection",
    #         text,
    #     )

    #     msg.exec()
    #     templatefiles, ok1 = QtWidgets.QFileDialog.getOpenFileNames(
    #         self, "QFileDialog.getOpenFileNames()", self._output_directory, "CSV file (*.csv)"
    #     )
    #     if ok1:
    #         templates = []
    #         for fnam in templatefiles:
    #             template = pd.read_csv(fnam, index_col=0)
    #             templates.append(template)

    #         corrscore_threshold, ok = QtWidgets.QInputDialog.getDouble(
    #             self,
    #             "Input Dialog",
    #             "Enter correlation threshold in (0-1):",
    #             decimals=2,
    #         )
    #         if corrscore_threshold > 1:
    #             corrscore_threshold = 1
    #         if corrscore_threshold < 0:
    #             corrscore_threshold = 0

    #         self._detectiondf_all = pd.DataFrame([])

    #         for i_block in range(len(self._input_data)):
    #             audiopath = self._input_data[i_block].filename

    #             if self.filename_timekey.text() == "":
    #                 self._input_data[i_block].date = dt.datetime(2000, 1, 1, 0, 0, 0)
    #             else:
    #                 try:
    #                     self._input_data[i_block].date = dt.datetime.strptime(
    #                         audiopath.split("/")[-1], self.filename_timekey.text()
    #                     )
    #                 except Exception:
    #                     self._input_data[i_block].date = dt.datetime(2000, 1, 1, 0, 0, 0)

    #             if self._input_data[i_block].start > 0:
    #                 secoffset = (
    #                     self._input_data[i_block].start
    #                     / self._input_data[i_block].audio_data.fs
    #                 )
    #                 self._input_data[i_block].date = self._input_data[i_block].date + pd.Timedelta(seconds=secoffset)

    #             self.read_soundfile(self._input_data[i_block])

    #             db_saturation = float(self.db_saturation.text())
    #             x = self._input_data[i_block].audio_data.x / 32767
    #             p = np.power(10, (db_saturation / 20)) * x  # convert data.signal to uPa

    #             fft_size = int(self.fft_size.currentText())
    #             fft_overlap = float(self.fft_overlap.currentText())
    #             # print(fft_size)
    #             # print(fft_overlap)

    #             self.compute_spectrogram(p, fft_size, fft_size * fft_overlap, self._input_data[i_block])

    #             if self._input_data[i_block].start > 0:
    #                 secoffset = (
    #                     self._input_data[i_block].start
    #                     / self._input_data[i_block].audio_data.fs
    #                 )
    #                 self._input_data[i_block].fft_data.t= self._input_data[i_block].fft_data.t+ secoffset

    #             # self.plotwindow_startsecond=0
    #             # self.plotwindow_length = self._input_data[i_block].fft_data.t.max()

    #             if templates[0].columns[0] == "Time_in_s":
    #                 # print(template)
    #                 offset_f = 10
    #                 offset_t = 0.5
    #                 # shape_f=template['Frequency_in_Hz'].values
    #                 # shape_t=template['Time_in_s'].values
    #                 # shape_t=shape_t-shape_t.min()
    #                 shape_f = np.array([])
    #                 shape_t_raw = np.array([])
    #                 for template in templates:
    #                     shape_f = np.concatenate(
    #                         [shape_f, template["Frequency_in_Hz"].values]
    #                     )
    #                     shape_t_raw = np.concatenate(
    #                         [shape_t_raw, template["Time_in_s"].values]
    #                     )
    #                 shape_t = shape_t_raw - shape_t_raw.min()

    #                 f_lim = [shape_f.min() - offset_f, shape_f.max() + offset_f]
    #                 k_length_seconds = shape_t.max() + offset_t * 2

    #                 # generate kernel
    #                 time_step = np.diff(self._input_data[i_block].fft_data.t)[0]

    #                 k_t = np.linspace(
    #                     0, k_length_seconds, int(k_length_seconds / time_step)
    #                 )
    #                 ix_f = np.where((self._input_data[i_block].fft_data.f >= f_lim[0]) & (self._input_data[i_block].fft_data.f <= f_lim[1]))[0]
    #                 k_f = self._input_data[i_block].fft_data.f[ix_f[0] : ix_f[-1]]
    #                 # k_f=np.linspace(f_lim[0],f_lim[1], int( (f_lim[1]-f_lim[0]) /f_step)  )

    #                 kk_t, kk_f = np.meshgrid(k_t, k_f)
    #                 kernel_background_db = 0
    #                 kernel_signal_db = 1
    #                 kernel = (
    #                     np.ones([k_f.shape[0], k_t.shape[0]]) * kernel_background_db
    #                 )
    #                 # find wich grid points are inside the shape
    #                 x, y = kk_t.flatten(), kk_f.flatten()
    #                 points = np.vstack((x, y)).T
    #                 # p = Path(list(zip(shape_t, shape_f))) # make a polygon
    #                 # grid = p.contains_points(points)
    #                 # mask = grid.reshape(kk_t.shape) # now you have a mask with points inside a polygon
    #                 # kernel[mask]=kernel_signal_db
    #                 for template in templates:
    #                     shf = template["Frequency_in_Hz"].values
    #                     st = template["Time_in_s"].values
    #                     st = st - shape_t_raw.min()
    #                     p = Path(list(zip(st, shf)))  # make a polygon
    #                     grid = p.contains_points(points)
    #                     kern = grid.reshape(
    #                         kk_t.shape
    #                     )  # now you have a mask with points inside a polygon
    #                     kernel[kern > 0] = kernel_signal_db

    #                 ix_f = np.where((self._input_data[i_block].fft_data.f >= f_lim[0]) & (self._input_data[i_block].fft_data.f <= f_lim[1]))[0]
    #                 spectrog = 10 * np.log10(self._input_data[i_block].fft_data.ssx[ix_f[0] : ix_f[-1], :])

    #                 result = match_template(spectrog, kernel)
    #                 corr_score = result[0, :]
    #                 t_score = np.linspace(
    #                     self._input_data[i_block].fft_data.t[int(kernel.shape[1] / 2)],
    #                     self._input_data[i_block].fft_data.t[-int(kernel.shape[1] / 2)],
    #                     corr_score.shape[0],
    #                 )

    #                 peaks_indices = find_peaks(corr_score, height=corrscore_threshold)[
    #                     0
    #                 ]

    #                 t1 = []
    #                 t2 = []
    #                 f1 = []
    #                 f2 = []
    #                 score = []

    #                 if len(peaks_indices) > 0:
    #                     t2_old = 0
    #                     for ixpeak in peaks_indices:
    #                         tstar = t_score[ixpeak] - k_length_seconds / 2 - offset_t
    #                         tend = t_score[ixpeak] + k_length_seconds / 2 - offset_t
    #                         # if tstar>t2_old:
    #                         t1.append(tstar)
    #                         t2.append(tend)
    #                         f1.append(f_lim[0] + offset_f)
    #                         f2.append(f_lim[1] - offset_f)
    #                         score.append(corr_score[ixpeak])
    #                         t2_old = tend
    #                     df = pd.DataFrame()
    #                     df["t-1"] = t1
    #                     df["t-2"] = t2
    #                     df["f-1"] = f1
    #                     df["f-2"] = f2
    #                     df["score"] = score

    #                     self._detectiondf = df.copy()
    #                     self._detectiondf["audiofilename"] = audiopath
    #                     self._detectiondf["threshold"] = corrscore_threshold
    #             else:  # image kernel
    #                 template = templates[0]

    #                 k_length_seconds = float(template.columns[-1]) - float(
    #                     template.columns[0]
    #                 )

    #                 f_lim = [int(template.index[0]), int(template.index[-1])]
    #                 ix_f = np.where((self._input_data[i_block].fft_data.f >= f_lim[0]) & (self._input_data[i_block].fft_data.f <= f_lim[1]))[0]
    #                 spectrog = 10 * np.log10(self._input_data[i_block].fft_data.ssx[ix_f[0] : ix_f[-1], :])
    #                 specgram_t_step = self._input_data[i_block].fft_data.t[1] - self._input_data[i_block].fft_data.t[0]
    #                 n_f = spectrog.shape[0]
    #                 n_t = int(k_length_seconds / specgram_t_step)

    #                 kernel = resize(template.values, [n_f, n_t])

    #                 result = match_template(spectrog, kernel)
    #                 corr_score = result[0, :]
    #                 t_score = np.linspace(
    #                     self._input_data[i_block].fft_data.t[int(kernel.shape[1] / 2)],
    #                     self._input_data[i_block].fft_data.t[-int(kernel.shape[1] / 2)],
    #                     corr_score.shape[0],
    #                 )

    #                 peaks_indices = find_peaks(corr_score, height=corrscore_threshold)[
    #                     0
    #                 ]

    #                 # print(corr_score)

    #                 t1 = []
    #                 t2 = []
    #                 f1 = []
    #                 f2 = []
    #                 score = []

    #                 if len(peaks_indices) > 0:
    #                     t2_old = 0
    #                     for ixpeak in peaks_indices:
    #                         tstar = t_score[ixpeak] - k_length_seconds / 2
    #                         tend = t_score[ixpeak] + k_length_seconds / 2
    #                         # if tstar>t2_old:
    #                         t1.append(tstar)
    #                         t2.append(tend)
    #                         f1.append(f_lim[0])
    #                         f2.append(f_lim[1])
    #                         score.append(corr_score[ixpeak])
    #                         t2_old = tend
    #                     df = pd.DataFrame()
    #                     df["t-1"] = t1
    #                     df["t-2"] = t2
    #                     df["f-1"] = f1
    #                     df["f-2"] = f2
    #                     df["score"] = score

    #                     self._detectiondf = df.copy()
    #                     self._detectiondf["audiofilename"] = audiopath
    #                     self._detectiondf["threshold"] = corrscore_threshold

    #             self._detectiondf_all = pd.concat(
    #                 [self._detectiondf_all, self._detectiondf]
    #             )
    #             self._detectiondf_all = self._detectiondf_all.reset_index(drop=True)

    #             print(self._detectiondf_all)

    #         self._detectiondf = self._detectiondf_all
    #         # self._detectiondf=self._detectiondf.reset_index(drop=True)
    #         self.generate_spectrogram()
    #         self.plot_spectrogram()
    #         print("done!!!")

    def automatic_detector_shapematching(self):
        # open template
        self._detectiondf = pd.DataFrame([])

        templatefiles, ok1 = QtWidgets.QFileDialog.getOpenFileNames(
            self, "QFileDialog.getOpenFileNames()", self._output_directory, "CSV file (*.csv)"
        )
        if ok1:
            self.notify_message(f"Opening drawing file {templatefiles} for shape matching")
            # print(template)
            # set db threshold
            db_threshold, ok = QtWidgets.QInputDialog.getInt(
                self, "Input Dialog", "Enter signal-to-noise threshold in dB:"
            )
            if ok:
                print(db_threshold)
                self._detectiondf = pd.DataFrame([])

                self.find_regions(db_threshold)
                self._detectiondf["score"] = np.zeros(len(self._detectiondf))

                for fnam in templatefiles:
                    template = pd.read_csv(fnam)
                    columns_to_check = ["Time_in_s", "Frequency_in_Hz", "audiofilename"]
                    found_columns = [ x in template.columns for x in columns_to_check ]
                    if not all(found_columns):
                        self.notify_message(f"Drawing file {fnam} malformed")
                        continue
                    score_new = self.match_bbox_and_iou(template)
                    ix_better = score_new > self._detectiondf["score"].values
                    # print(score_new)
                    # print( self._detectiondf['score'].values)
                    # print( 'better:' )
                    # print( ix_better )

                    self._detectiondf.loc[ix_better, "score"] = score_new[ix_better]
                    # print(self._detectiondf['score'])

                ixdel = np.where(self._detectiondf["score"] < 0.01)[0]
                self._detectiondf = self._detectiondf.drop(ixdel)
                self._detectiondf = self._detectiondf.reset_index(drop=True)
                self._detectiondf["audiofilename"] = self._input_data[self._filepointer].filename
                self._detectiondf["threshold"] = db_threshold

                print(self._detectiondf)

                # plot results
                self.plot_spectrogram()

    # def automatic_detector_shapematching_allfiles(self):
    #     text = (
    #         "Are you sure you want to run the detector over "
    #         + str(len(self._input_data))
    #         + " ?"
    #     )

    #     msg = QtWidgets.QMessageBox.information(
    #         self,
    #         "Detection",
    #         text,
    #     )
    #     msg.exec()
    #     templatefiles, ok1 = QtWidgets.QFileDialog.getOpenFileNames(
    #         self, "QFileDialog.getOpenFileNames()", self._output_directory, "CSV file (*.csv)"
    #     )
    #     # template=pd.read_csv(templatefile)
    #     db_threshold, ok = QtWidgets.QInputDialog.getInt(
    #         self, "Input Dialog", "Enter signal-to-noise threshold in dB:"
    #     )

    #     self._detectiondf_all = pd.DataFrame([])

    #     for i_block in range(len(self._input_data)):
    #         audiopath = self._input_data[i_block].filename

    #         if self.filename_timekey.text() == "":
    #             self._input_data[self._filepointer].date = dt.datetime(2000, 1, 1, 0, 0, 0)
    #         else:
    #             try:
    #                 self._input_data[self._filepointer].date = dt.datetime.strptime(
    #                     audiopath.split("/")[-1], self.filename_timekey.text()
    #                 )
    #             except Exception:
    #                 self._input_data[self._filepointer].date = dt.datetime(2000, 1, 1, 0, 0, 0)

    #         self.read_soundfile(self._input_data[i_block])

    #         db_saturation = float(self.db_saturation.text())
    #         x = self._input_data[i_block].audio_data.x / 32767
    #         p = np.power(10, (db_saturation / 20)) * x  # convert data.signal to uPa

    #         fft_size = int(self.fft_size.currentText())
    #         fft_overlap = float(self.fft_overlap.currentText())

    #         self.compute_spectrogram(p, fft_size, fft_size * fft_overlap, self._input_data[i_block])

    #         if self._input_data[i_block].start > 0:
    #             secoffset = (
    #                 self._input_data[i_block].start / self._input_data[i_block].audio_data.fs
    #             )
    #             self._input_data[i_block].fft_data.t= self._input_data[i_block].fft_data.t+ secoffset

    #         self.plotwindow_startsecond = 0
    #         self.plotwindow_length = self._input_data[i_block].fft_data.t.max()

    #         self._detectiondf = pd.DataFrame([])

    #         self.find_regions(db_threshold)
    #         # match_bbox_and_iou(template)
    #         self._detectiondf["score"] = np.zeros(len(self._detectiondf))
    #         for fnam in templatefiles:
    #             template = pd.read_csv(fnam, index_col=0)
    #             score_new = self.match_bbox_and_iou(template)
    #             ix_better = score_new > self._detectiondf["score"].values
    #             self._detectiondf.loc[ix_better, "score"] = score_new[ix_better]

    #         ixdel = np.where(self._detectiondf["score"] < 0.01)[0]
    #         self._detectiondf = self._detectiondf.drop(ixdel)
    #         self._detectiondf = self._detectiondf.reset_index(drop=True)
    #         self._detectiondf["audiofilename"] = audiopath
    #         self._detectiondf["threshold"] = db_threshold

    #         self._detectiondf_all = pd.concat([self._detectiondf_all, self._detectiondf])
    #         self._detectiondf_all = self._detectiondf_all.reset_index(drop=True)

    #         print(self._detectiondf_all)

    #     self._detectiondf = self._detectiondf_all
    #     # self._detectiondf=self._detectiondf.reset_index(drop=True)
    #     print("done!!!")

    def export_automatic_detector(self):
        if self._detectiondf is None or self._detectiondf.empty:
            return

        if self._detectiondf.shape[0] > 0:
            savename = QtWidgets.QFileDialog.getSaveFileName(
                self,
                "QFileDialog.getSaveFileName()",
                self._output_directory,
                "csv files (*.csv)",
            )
            print("location is:" + savename[0])
            if len(savename[0]) > 0:
                self._detectiondf.to_csv(savename[0])

    def open_output_directory(self, directory: str):
        self.outputview_model = QFileSystemModel()
        self.outputview.setModel(self.outputview_model)
        self.outputview_model.setRootPath(
            QDir.rootPath()
        )  # Set the root path to the system root
        self.outputview.setRootIndex(self.outputview_model.index(directory))
        self._output_directory = directory
        self.settings.setValue("output_directory", directory)

    def set_user_library(self):
        directory = QtWidgets.QFileDialog.getExistingDirectory(
            self, directory=self.settings.value("library_directory", os.getcwd())
        )
        if directory is not None and directory:
            self.open_library(os.path.abspath(directory))

    def set_output_directory(self):
        directory = QtWidgets.QFileDialog.getExistingDirectory(
            self, directory=self.settings.value("output_directory", os.getcwd())
        )
        if directory is not None and directory:
            self.open_output_directory(os.path.abspath(directory))

    def open_library(self, directory: str):
        self.filesview_model = QFileSystemModel()
        self.filesview.setModel(self.filesview_model)
        self.filesview_model.setRootPath(
            QDir.rootPath()
        )  # Set the root path to the system root
        self.filesview.setRootIndex(self.filesview_model.index(directory))
        self.library_directory = directory
        self.settings.setValue("library_directory", directory)

    def on_file_double_clicked(self, index: QModelIndex):
        full_path = self.filesview_model.filePath(index)
        if os.path.isfile(full_path):
            self.notify_message(f"Opening: {full_path}")
            self.open_audio_file(full_path)
            self.plot_spectrogram()

    def open_audio_file(self, path: str):
        self.plotwindow_startsecond = float(self.t_length.text())

        self._detectiondf = pd.DataFrame([])

        is_already_loaded = list(filter(lambda x: x.filename == path, self._input_data))
        if len(is_already_loaded) == 0:
            try:
                a = sf.info(str(path))
            except Exception as e:
                self.notify_message(f"{e}")
            else:
                if a.duration < MAX_LENGHT_SEC:
                    ip = InputData(
                        filename=path,
                        start=0,
                        end=a.frames,
                    )
                    if self.filename_timekey.text():
                        try:
                            ip.date = dt.datetime.strptime(
                                os.path.basename(path),
                                self.filename_timekey.text(),
                            )
                        except Exception as e:
                            self.notify_message(f"Error parsing file timestamp: {e}")
                    self._input_data.append(ip)
                else:
                    s = 0
                    while s < a.frames:
                        e = s + MAX_LENGHT_SEC * a.samplerate
                        ip = InputData(
                            filename=str(path),
                            start=s,
                            end=e,
                        )
                        self._input_data.append(ip)
                        s = s + MAX_LENGHT_SEC * a.samplerate

                self.plotwindow_startsecond = 0
                self._filepointer = len(self._input_data) - 1
                self.generate_spectrogram()
        else:
            for i in range(len(self._input_data)):
                if self._input_data[i].filename == path:
                    self._filepointer = i
                    break

    def read_soundfile(self, input_data: InputData, dtype: str = "int16"):
        if input_data.audio_data is None:
            x, fs = sf.read(
                input_data.filename,
                start=input_data.start,
                stop=input_data.end,
                dtype=dtype,
            )
            if len(x.shape) > 1:
                if np.shape(x)[1] > 1:
                    x = x[:, 0]
            sample = AudioSample(x=x, fs=fs)
            input_data.audio_data = sample

    def compute_spectrogram(self, p, nperseg, noverlap, input_data:InputData):
        if input_data.fft_data is None:
            try:
                f, t, ssx = signal.spectrogram(
                    p,
                    input_data.audio_data.fs,
                    window="hamming",
                    nperseg=nperseg,
                    noverlap=noverlap,
                )
            except Exception as e:
                self.notify_message(f"Could not generate spectrogram: {e}")
            else:
                ffts = FFTSample(
                    ssx=ssx,
                    f=f,
                    t=t,
                )
                input_data.fft_data = ffts

    def generate_spectrogram(self):
        if self._filepointer >= 0:
            # if self.filename_timekey.text()=='':
            #     self._input_data[self._filepointer].date= dt.datetime(1,1,1,0,0,0)
            # else:
            #     self._input_data[self._filepointer].date= dt.datetime.strptime( audiopath.split('/')[-1], self.filename_timekey.text() )

            # if audiopath[-4:]=='.wav':

            self.read_soundfile(self._input_data[self._filepointer])

            if self._input_data[self._filepointer].start > 0:
                secoffset = (
                    self._input_data[self._filepointer].start
                    / self._input_data[self._filepointer].audio_data.fs
                )
                self._input_data[self._filepointer].date = self._input_data[self._filepointer].date + pd.Timedelta(seconds=secoffset)

            # factor=60
            # x=signal.decimate(x,factor,ftype='fir')

            db_saturation = float(self.db_saturation.text())
            x = self._input_data[self._filepointer].audio_data.x / 32767
            p = np.power(10, (db_saturation / 20)) * x  # convert data.signal to uPa

            fft_size = int(self.fft_size.currentText())
            fft_overlap = float(self.fft_overlap.currentText())
            nperseg = fft_size
            noverlap = int(fft_size * fft_overlap)
            if noverlap > nperseg:
                nperseg = fft_size
                noverlap = None

            self.compute_spectrogram(p, nperseg, noverlap, self._input_data[self._filepointer])
            # self._input_data[self._filepointer].fft_data.t=self._input_data[self._filepointer].date +  pd.to_timedelta( t  , unit='s')
            if self._input_data[self._filepointer].start > 0:
                secoffset = (
                    self._input_data[self._filepointer].start
                    / self._input_data[self._filepointer].audio_data.fs
                )
                self._input_data[self._filepointer].fft_data.t= self._input_data[self._filepointer].fft_data.t+ secoffset
            # print(self._input_data[self._filepointer].fft_data.t)


    def plot_annotations(self):
        if len(self._input_data) == 0:
            return

        # plot annotations
        if self._input_data[self._filepointer].annotations is not None and not self._input_data[self._filepointer].annotations.empty:
            ix = (
                (
                    self._input_data[self._filepointer].annotations["t1"]
                    > (
                        np.array(self._input_data[self._filepointer].date).astype("datetime64[ns]")
                        + pd.Timedelta(self.plotwindow_startsecond, unit="s")
                    )
                )
                & (
                    self._input_data[self._filepointer].annotations["t1"]
                    < (
                        np.array(self._input_data[self._filepointer].date).astype("datetime64[ns]")
                        + pd.Timedelta(
                            self.plotwindow_startsecond + self.plotwindow_length,
                            unit="s",
                        )
                    )
                )
                & (self._input_data[self._filepointer].annotations["audiofilename"] == self._input_data[self._filepointer].filename)
            )
            if np.sum(ix) > 0:
                ix = np.where(ix)[0]
                for ix_x in ix:
                    a = pd.DataFrame([self._input_data[self._filepointer].annotations.iloc[ix_x, :]])
                    self.plot_annotation_box(a)

    def plot_annotation_box(self, annotation_row):
        print(annotation_row)
        x1 = annotation_row.iloc[0, 0]
        x2 = annotation_row.iloc[0, 1]
        y1 = annotation_row.iloc[0, 2]
        y2 = annotation_row.iloc[0, 3]
        c_label = annotation_row.iloc[0, 4]

        xt = pd.Series([x1, x2])

        # print(np.dtype(np.array(self._input_data[self._filepointer].date).astype('datetime64[ns]') ))
        tt = xt - np.array(self._input_data[self._filepointer].date).astype("datetime64[ns]")
        xt = tt.dt.seconds + tt.dt.microseconds / 10**6
        x1 = xt[0]
        x2 = xt[1]

        # tt=x1 - np.array(self._input_data[self._filepointer].date).astype('datetime64[ns]')
        # x1=tt.dt.seconds + tt.dt.microseconds/10**6
        # tt=x2 - np.array(self._input_data[self._filepointer].date).astype('datetime64[ns]')
        # x2=tt.dt.seconds + tt.dt.microseconds/10**6

        line_x = [x2, x1, x1, x2, x2]
        line_y = [y1, y1, y2, y2, y1]

        xmin = np.min([x1, x2])
        ymax = np.max([y1, y2])

        self.canvas.axes.plot(line_x, line_y, "-b", linewidth=0.75)
        self.canvas.axes.text(xmin, ymax, c_label, size=8)
        self.canvas.draw()

    def plot_spectrogram(self):
        if len(self._input_data) == 0:
            return

        if self._input_data[self._filepointer].fft_data is None \
            or self._input_data[self._filepointer].audio_data is None:

            self.canvas.fig.clf()
            self.canvas.axes = self.canvas.fig.add_subplot(111)
            self.canvas.axes.set_title(os.path.basename(self._input_data[self._filepointer].filename))
            self.canvas.fig.tight_layout()
            self.canvas.draw()
            return

        # self.canvas = MplCanvas(self, width=5, height=4, dpi=100)
        # self.setCentralWidget(self.canvas)
        self.canvas.fig.clf()
        self.canvas.axes = self.canvas.fig.add_subplot(111)
        # self.canvas.axes.cla()

        if self.t_length.text() == "":
            self.plotwindow_length = self._input_data[self._filepointer].fft_data.t[-1]
            self.plotwindow_startsecond = self._input_data[self._filepointer].fft_data.t[0]
        else:
            self.plotwindow_length = float(self.t_length.text())
            if self._input_data[self._filepointer].fft_data.t[-1] < self.plotwindow_length:
                self.plotwindow_startsecond = self._input_data[self._filepointer].fft_data.t[0]
                self.plotwindow_length = self._input_data[self._filepointer].fft_data.t[-1]

        y1 = int(self.f_min.text())
        y2 = int(self.f_max.text())
        if y2 > (self._input_data[self._filepointer].audio_data.fs / 2):
            y2 = self._input_data[self._filepointer].audio_data.fs / 2
        t1 = self.plotwindow_startsecond
        t2 = self.plotwindow_startsecond + self.plotwindow_length

        # if self.t_length.text=='':
        #     t2=self._input_data[self._filepointer].fft_data.t[-1]
        # else:
        #     if self._input_data[self._filepointer].fft_data.t[-1]<float(self.t_length.text()):
        #         t2=self._input_data[self._filepointer].fft_data.t[-1]
        #     else:
        #         t2=self.plotwindow_startsecond+self.plotwindow_length

        # tt,ff=np.meshgrid(self._input_data[self._filepointer].fft_data.t,self._input_data[self._filepointer].fft_data.f)
        # ix_time=(tt>=self.plotwindow_startsecond) & (tt<(self.plotwindow_startsecond+self.plotwindow_length))
        # ix_f=(ff>=y1) & (ff<y2)
        # plotsxx=self._input_data[self._filepointer].fft_data.ssx[ ix_f & ix_time]
        ix_time = np.where((self._input_data[self._filepointer].fft_data.t>= t1) & (self._input_data[self._filepointer].fft_data.t< t2))[0]
        ix_f = np.where((self._input_data[self._filepointer].fft_data.f >= y1) & (self._input_data[self._filepointer].fft_data.f < y2))[0]
        # print(ix_time.shape)
        # print([self._input_data[self._filepointer].fft_data.t,t1,t2])
        plotsxx = self._input_data[self._filepointer].fft_data.ssx[
            int(ix_f[0]) : int(ix_f[-1]), int(ix_time[0]) : int(ix_time[-1])
        ]
        plotsxx_db = 10 * np.log10(plotsxx)

        if self.checkbox_background.isChecked():
            spec_mean = np.median(plotsxx_db, axis=1)
            sxx_background = np.transpose(
                np.broadcast_to(spec_mean, np.transpose(plotsxx_db).shape)
            )
            plotsxx_db = plotsxx_db - sxx_background
            plotsxx_db = plotsxx_db - np.min(plotsxx_db.flatten())
        # print(plotsxx.shape)

        # img=self.canvas.axes.pcolormesh(self._input_data[self._filepointer].fft_data.t, self._input_data[self._filepointer].fft_data.f, 10*np.log10(self._input_data[self._filepointer].fft_data.ssx) ,cmap='plasma')
        colormap_plot = self.colormap_plot.currentText()
        img = self.canvas.axes.imshow(
            plotsxx_db,
            aspect="auto",
            cmap=colormap_plot,
            origin="lower",
            extent=[t1, t2, y1, y2],
        )

        # img=self.canvas.axes.pcolormesh(self._input_data[self._filepointer].fft_data.t[ int(ix_time[0]):int(ix_time[-1])], self._input_data[self._filepointer].fft_data.f[int(ix_f[0]):int(ix_f[-1])], 10*np.log10(plotsxx) , shading='flat',cmap='plasma')

        self.canvas.axes.set_ylabel("Frequency [Hz]")
        self.canvas.axes.set_xlabel("Time [sec]")
        if self.checkbox_logscale.isChecked():
            self.canvas.axes.set_yscale("log")
        else:
            self.canvas.axes.set_yscale("linear")

        if self._input_data[self._filepointer].date != dt.datetime(1970, 1, 1, 0, 0):
            self.canvas.axes.set_title(self._input_data[self._filepointer].date)
        else:
            self.canvas.axes.set_title(os.path.basename(self._input_data[self._filepointer].filename))


        # img.set_clim([ 40 ,10*np.log10( np.max(np.array(plotsxx).ravel() )) ] )
        clims = img.get_clim()
        if (self.db_vmin.text() == "") & (self.db_vmax.text() != ""):
            img.set_clim([clims[0], float(self.db_vmax.text())])
        if (self.db_vmin.text() != "") & (self.db_vmax.text() == ""):
            img.set_clim([float(self.db_vmin.text()), clims[1]])
        if (self.db_vmin.text() != "") & (self.db_vmax.text() != ""):
            img.set_clim([float(self.db_vmin.text()), float(self.db_vmax.text())])

        self.canvas.fig.colorbar(img, label="PSD [dB re $1 \ \mu Pa \ Hz^{-1}$]")

        # print(self._input_data[self._filepointer].date)
        # print(self.call_time)

        # plot detections
        cmap = plt.cm.get_cmap("cool")
        if self._detectiondf.shape[0] > 0:
            for i in range(self._detectiondf.shape[0]):
                insidewindow = (
                    (self._detectiondf.loc[i, "t-1"] > self.plotwindow_startsecond)
                    & (
                        self._detectiondf.loc[i, "t-2"]
                        < (self.plotwindow_startsecond + self.plotwindow_length)
                    )
                    & (
                        self._detectiondf.loc[i, "audiofilename"]
                        == self._input_data[self._filepointer].filename
                    )
                )

                scoremin = self._detectiondf["score"].min()
                scoremax = self._detectiondf["score"].max()

                if (self._detectiondf.loc[i, "score"] >= 0.01) & insidewindow:
                    xx1 = self._detectiondf.loc[i, "t-1"]
                    xx2 = self._detectiondf.loc[i, "t-2"]
                    yy1 = self._detectiondf.loc[i, "f-1"]
                    yy2 = self._detectiondf.loc[i, "f-2"]
                    scorelabel = str(np.round(self._detectiondf.loc[i, "score"], 2))
                    snorm = (self._detectiondf.loc[i, "score"] - scoremin) / (
                        scoremax - scoremin
                    )
                    scorecolor = cmap(snorm)

                    line_x = [xx2, xx1, xx1, xx2, xx2]
                    line_y = [yy1, yy1, yy2, yy2, yy1]

                    xmin = np.min([xx1, xx2])
                    ymax = np.max([yy1, yy2])
                    self.canvas.axes.plot(
                        line_x, line_y, "-", color=scorecolor, linewidth=0.75
                    )
                    self.canvas.axes.text(
                        xmin, ymax, scorelabel, size=8, color=scorecolor
                    )

        self.canvas.axes.set_ylim([y1, y2])
        self.canvas.axes.set_xlim([t1, t2])

        self.canvas.fig.tight_layout()
        self.canvas.draw()

    def plot_spectrogram_threshold(self):
        if self._filepointer >= 0:
            db_threshold, ok = QtWidgets.QInputDialog.getInt(
                self, "Input Dialog", "Enter signal-to-noise threshold in dB:"
            )
            self.find_regions(db_threshold)
            self._detectiondf = pd.DataFrame([])

            # plot_spectrogram()
            # self.canvas.axes2 = self.canvas.axes.twiny()

            # self.canvas.axes2.contour( self.region_labels>0 , [0.5] , color='g')

            self.canvas.fig.clf()
            self.canvas.axes = self.canvas.fig.add_subplot(111)

            self.canvas.axes.set_ylabel("Frequency [Hz]")
            # self.canvas.axes.set_xlabel('Time [sec]')
            if self.checkbox_logscale.isChecked():
                self.canvas.axes.set_yscale("log")
            else:
                self.canvas.axes.set_yscale("linear")

            img = self.canvas.axes.imshow(
                self.region_labels > 0, aspect="auto", cmap="gist_yarg", origin="lower"
            )

            self.canvas.fig.colorbar(img)

            self.canvas.fig.tight_layout()
            self.canvas.draw()

    def export_zoomed_sgram_as_csv(self):
        if self._filepointer >= 0:

            if self._input_data[self._filepointer].fft_data is None \
                or self._input_data[self._filepointer].audio_data is None:
                self.notify_message(f"No data to save for file {self._input_data[self._filepointer].filename}")
                return
            # filter out background
            spectrog = 10 * np.log10(self._input_data[self._filepointer].fft_data.ssx)

            msg = QtWidgets.QMessageBox.information(
                self,
                "Export",
                "Remove background",
            )
            returnValue = msg.exec()

            if returnValue == QtWidgets.QMessageBox.StandardButton.Yes:
                rectime = pd.to_timedelta(self._input_data[self._filepointer].fft_data.t, "s")
                spg = pd.DataFrame(np.transpose(spectrog), index=rectime)
                bg = spg.resample("3min").mean().copy()
                bg = bg.resample("1s").interpolate(method="time")
                bg = bg.reindex(rectime, method="nearest")
                background = np.transpose(bg.values)
                z = spectrog - background
            else:
                z = spectrog

            self.f_limits = self.canvas.axes.get_ylim()
            self.t_limits = self.canvas.axes.get_xlim()
            y1 = int(self.f_limits[0])
            y2 = int(self.f_limits[1])
            t1 = self.t_limits[0]
            t2 = self.t_limits[1]

            ix_time = np.where((self._input_data[self._filepointer].fft_data.t>= t1) & (self._input_data[self._filepointer].fft_data.t< t2))[0]
            ix_f = np.where((self._input_data[self._filepointer].fft_data.f >= y1) & (self._input_data[self._filepointer].fft_data.f < y2))[0]
            # print(ix_time.shape)
            # print(ix_f.shape)
            plotsxx_db = z[
                int(ix_f[0]) : int(ix_f[-1]), int(ix_time[0]) : int(ix_time[-1])
            ]

            sgram = pd.DataFrame(
                data=plotsxx_db, index=self._input_data[self._filepointer].fft_data.f[ix_f[:-1]], columns=self._input_data[self._filepointer].fft_data.t[ix_time[:-1]]
            )
            print(sgram)

            savename = QtWidgets.QFileDialog.getSaveFileName(
                self, self._output_directory, "csv files (*.csv)"
            )
            if len(savename[0]) > 0:
                if savename[-4:] != ".csv":
                    savename = savename[0] + ".csv"
                sgram.to_csv(savename)

    def box_select_callback(self, eclick, erelease):
        x1, y1 = eclick.xdata, eclick.ydata
        x2, y2 = erelease.xdata, erelease.ydata

        x1 = self._input_data[self._filepointer].date + pd.to_timedelta(x1, unit="s")
        x2 = self._input_data[self._filepointer].date + pd.to_timedelta(x2, unit="s")

        # sort to increasing values
        t1 = np.min([x1, x2])
        t2 = np.max([x1, x2])
        f1 = np.min([y1, y2])
        f2 = np.max([y1, y2])

        if self.bg.checkedId() == -1:
            c_label = ""
        else:
            c_label = eval("self.an_" + str(self.bg.checkedId()) + ".text()")

        # a=pd.DataFrame(columns=['t1','t2','f1','f2','label'])
        # a.iloc[0,:]=np.array([x1,x2,y1,y2,c_label ])
        a = pd.DataFrame(
            {
                "t1": pd.Series(t1, dtype="datetime64[ns]"),
                "t2": pd.Series(t2, dtype="datetime64[ns]"),
                "f1": pd.Series(f1, dtype="float"),
                "f2": pd.Series(f2, dtype="float"),
                "label": pd.Series(c_label, dtype="object"),
                "audiofilename": self._input_data[self._filepointer].filename,
            }
        )

        # a=pd.DataFrame(data=[ [x1,x2,y1,y2,c_label ] ],columns=['t1','t2','f1','f2','label'])
        # print('a:')
        # print(a.dtypes)
        # self._input_data[self._filepointer].annotations.append(a, ignore_index = True)
        self._input_data[self._filepointer].annotations = pd.concat([self._input_data[self._filepointer].annotations, a], ignore_index=True)

        # print(self._input_data[self._filepointer].annotations.dtypes)
        self.plot_annotation_box(a.iloc[:1])

    def onclick_annotate(self, event):
        if event.button == 3:
            self._input_data[self._filepointer].annotations = self._input_data[self._filepointer].annotations.head(-1)
            self.plot_spectrogram()
            self.plot_annotations()

    def plot_next_spectro(self):
        if len(self._input_data) == 0 or self._filepointer == len(self._input_data)-1:
            return

        self._filepointer = self._filepointer + 1
        self.generate_spectrogram()
        self.plot_spectrogram()
        self.plot_annotations()

    def plot_previous_spectro(self):
        if len(self._input_data) == 0 or self._filepointer == 0:
            return

        self._filepointer = self._filepointer - 1
        self.generate_spectrogram()
        self.plot_spectrogram()
        self.plot_annotations()

    def new_fft_size_selected(self):
        self.generate_spectrogram()
        self.plot_spectrogram()

    def load_annotations(self):
        file = QtWidgets.QFileDialog.getOpenFileNames(
            self, directory=self._output_directory
        )

        if not file[0]:
            self.notify_message("Annotation loading aborted")
            return

        self.load_annotations_files(file[0])

    def load_annotations_files(self, files:list):
        for annotation_file in files:
            try:
                annotations = pd.read_csv(
                        annotation_file,
                        parse_dates=["t1", "t2"],
                        dtype = {
                            "f1": float,
                            "f2": float,
                            "label": object,
                            "audiofilename": object,
                        }
                    )
            except Exception as e:
                self.notify_message(f"Annotation file {annotation_file} malformed: {e}")
                continue
            else:
                if annotations.empty:
                    self.notify_message(f"Annotation file {annotation_file} empty")
                    return

                columns_to_check = ["t1", "t2", "f1", "f2", "label", "audiofilename"]
                found_columns = [ x in annotations.columns for x in columns_to_check ]
                if not all(found_columns):
                    self.notify_message(f"Annotation file {annotation_file} malformed")
                    return

                self.notify_message(f"Opening annotations file {annotation_file}")
                self.open_audio_file(annotations["audiofilename"][0])
                self.plot_spectrogram()
                self._input_data[self._filepointer].annotations = annotations
                self._input_data[self._filepointer].annotations_file = annotation_file
                self.plot_annotations()

    def func_annotate_save(self):
        for layout in [self.top2_layout, self.top3_layout, self.sidepanel_layout, self.plot_layout]:
            self.set_enabled_layout(layout, True)
        if self._input_data[self._filepointer].annotations_file is None:
            ddate = dt.datetime.now().strftime("%Y_%m_%d__%H_%M_%S")
            savename = os.path.join(self._output_directory, f"annotations_{ddate}.csv")
            self._input_data[self._filepointer].annotations_file = savename
        self.canvas.fig.canvas.mpl_disconnect(self.mpl_action_annotate)
        self.notify_message(f"Saving annotations to {self._input_data[self._filepointer].annotations_file}")
        self._input_data[self._filepointer].annotations.to_csv(self._input_data[self._filepointer].annotations_file, index=False)
        self.toggle_selector = None
        del self.toggle_selector
        self.plot_spectrogram()
        self.plot_annotations()

    def func_annotate_abort(self):
        for layout in [self.top2_layout, self.top3_layout, self.sidepanel_layout, self.plot_layout]:
            self.set_enabled_layout(layout, True)
        self.canvas.fig.canvas.mpl_disconnect(self.mpl_action_annotate)
        self.toggle_selector = None
        del self.toggle_selector
        self.notify_message("Aborting annotation")
        self.plot_spectrogram()
        self.plot_annotations()

    # def func_logging(self):
    #     if self.checkbox_log.isChecked():
    #         print("logging")

    #         msg = QtWidgets.QMessageBox.information(
    #             self,
    #             "Logging",
    #             "Overwrite existing logs files ?",
    #         )
    #         returnValue = msg.exec()
    #         if returnValue == QtWidgets.QMessageBox.StandardButton.No:
    #             ix_delete = []
    #             i = 0
    #             for fn in self.filenames:
    #                 logpath = fn[:-4] + "_log.csv"
    #                 if os.path.isfile(logpath):
    #                     ix_delete.append(i)
    #                 i = i + 1
    #             self.filenames = np.delete(self.filenames, ix_delete)

    def export_all_spectrograms(self):
        for i in range(self._filepointer + 1):
            audiopath = self._input_data[i].filename

            self.read_soundfile(self._input_data[i])
            db_saturation = float(self.db_saturation.text())
            x = self._input_data[i].audio_data.x / 32767
            p = np.power(10, (db_saturation / 20)) * x  # convert data.signal to uPa

            fft_size = int(self.fft_size.currentText())
            fft_overlap = float(self.fft_overlap.currentText())

            nperseg = fft_size
            noverlap = int(fft_size * fft_overlap)
            if noverlap > nperseg:
                nperseg = fft_size
                noverlap = None

            try:
                self._input_data[i].fft_data.f, self._input_data[i].fft_data.t, self._input_data[i].fft_data.ssx = signal.spectrogram(
                    p,
                    self._input_data[i].audio_data.fs,
                    window="hamming",
                    nperseg=nperseg,
                    noverlap=noverlap,
                )
            except Exception as e:
                self.notify_message(f"Could not export spectrogram of {self._input_data[i].filename}: {e}")
                continue
            else:
                self.plotwindow_startsecond = 0

                self.plot_spectrogram()
                self.canvas.axes.set_title(os.path.basename(audiopath))
                output_path = os.path.join(
                    self._output_directory, os.path.basename(audiopath) + ".jpg"
                )
                self.notify_message(f"Saving spectrogram to {output_path}")
                self.canvas.fig.savefig(output_path, dpi=150)

    def plot_drawing(self):
        if self._filepointer >= 0:
            if self._input_data[self._filepointer].drawing is not None and not self._input_data[self._filepointer].drawing.empty:
                self.canvas.axes.plot(self._input_data[self._filepointer].drawing["Time_in_s"], self._input_data[self._filepointer].drawing["Frequency_in_Hz"], ".-g")
                self.canvas.draw()

    def onclick_draw(self, event):
        # print('%s click: button=%d, x=%d, y=%d, xdata=%f, ydata=%f' %
        #       ('double' if event.dblclick else 'single', event.button,
        #        event.x, event.y, event.xdata, event.ydata))
        if event.button == 1 & event.dblclick:
            new_point = pd.DataFrame.from_dict({"Time_in_s": [event.xdata], "Frequency_in_Hz": [event.ydata], "audiofilename":[self._input_data[self._filepointer].filename]})
            if self._input_data[self._filepointer].drawing.empty:
                self._input_data[self._filepointer].drawing = new_point
            else:
                self._input_data[self._filepointer].drawing = pd.concat([self._input_data[self._filepointer].drawing, new_point], ignore_index=True)
            self.f_limits = self.canvas.axes.get_ylim()
            self.t_limits = self.canvas.axes.get_xlim()

            # line = self.line_2.pop(0)
            self.line_2 = self.canvas.axes.plot(self._input_data[self._filepointer].drawing["Time_in_s"], self._input_data[self._filepointer].drawing["Frequency_in_Hz"], ".-g")
            self.canvas.draw()
            self.notify_message(f"Added point {event.xdata}, {event.ydata}")

        if event.button == 3:
            self._input_data[self._filepointer].drawing.drop(self._input_data[self._filepointer].drawing.tail(1), inplace=True)
            self.f_limits = self.canvas.axes.get_ylim()
            self.t_limits = self.canvas.axes.get_xlim()
            line = self.line_2.pop(0)
            line.remove()
            if not self._input_data[self._filepointer].drawing.empty:
                self.line_2 = self.canvas.axes.plot(self._input_data[self._filepointer].drawing["Time_in_s"], self._input_data[self._filepointer].drawing["Frequency_in_Hz"], ".-g")
                self.canvas.draw()
                self.notify_message("Removed latest point")

    def func_draw_shape_abort(self):
        for layout in [self.top2_layout, self.top3_layout, self.sidepanel_layout, self.plot_layout]:
            self.set_enabled_layout(layout, True)

        self.canvas.fig.canvas.mpl_disconnect(self.mpl_action_draw)
        self.plot_spectrogram()
        self.notify_message("Aborting drawing")

    def func_draw_shape_save(self):
        for layout in [self.top2_layout, self.top3_layout, self.sidepanel_layout, self.plot_layout]:
            self.set_enabled_layout(layout, True)

        self.canvas.fig.canvas.mpl_disconnect(self.mpl_action_draw)
        self.plot_spectrogram()

        if not self._input_data[self._filepointer].drawing.empty:
            ddate = dt.datetime.now().strftime("%Y_%m_%d__%H_%M_%S")
            savename = os.path.join(self._output_directory, f"draw_{ddate}.csv")
            self._input_data[self._filepointer].drawing.to_csv(savename, index=False)
            self.notify_message(f"Saving drawing to {savename}")
        else:
            self.notify_message("No drawing to save")


    def set_enabled_layout(self, layout, value):
        for i in range(layout.count()):
            item = layout.itemAt(i)
            widget = item.widget()
            if widget is not None:
                widget.setEnabled(value)
            else:
                sub_layout = item.layout()
                if sub_layout is not None:
                    self.set_enabled_layout(sub_layout, value)

    def func_draw_shape(self, b:bool):
        if len(self._input_data) == 0:
            return
        for layout in [self.top2_layout, self.top3_layout, self.sidepanel_layout, self.plot_layout]:
            self.set_enabled_layout(layout, False)
        self.canvas.setEnabled(True)
        self.notify_message("Drawing mode")
        if self._input_data[self._filepointer].drawing is None:
            self._input_data[self._filepointer].drawing = pd.DataFrame(columns=["Time_in_s", "Frequency_in_Hz", "audiofilename"])

        self.f_limits = self.canvas.axes.get_ylim()
        self.t_limits = self.canvas.axes.get_xlim()
        self.mpl_action_draw = self.canvas.fig.canvas.mpl_connect("button_press_event", self.onclick_draw)
        if hasattr(self, "mpl_action_annotate"):
            self.canvas.fig.canvas.mpl_disconnect(self.mpl_action_annotate)
        if not hasattr(self, "exitaction"):
            self.exitaction = QShortcut(QtCore.Qt.Key.Key_Return, self)
        else:
            self.exitaction.activated.disconnect()

        self.exitaction.activated.connect(self.func_draw_shape_save)
        if not hasattr(self, "abortaction"):
            self.abortaction = QShortcut(QtCore.Qt.Key.Key_Escape, self)
        else:
            self.abortaction.activated.disconnect()

        self.abortaction.activated.connect(self.func_draw_shape_abort)

        if not self._input_data[self._filepointer].drawing.empty:
            self.line_2 = self.canvas.axes.plot(self._input_data[self._filepointer].drawing["Time_in_s"], self._input_data[self._filepointer].drawing["Frequency_in_Hz"], ".-g")
        self.plot_drawing()

    def load_drawing(self):
        file = QtWidgets.QFileDialog.getOpenFileName(
            self, directory=self._output_directory
        )

        if not file[0]:
            self.notify_message("Drawing loading aborted")
            return
        self.load_drawing_file(file[0])

    def load_drawing_file(self, path:str):
        try:
            drawing = pd.read_csv(
                    path,
                    dtype = {
                        "Time_in_s": float,
                        "Frequency_in_Hz": float,
                        "audiofilename": object,
                    }
                )
        except Exception as e:
            self.notify_message(f"Drawing file {path} malformed: {e}")
        else:

            if drawing.empty:
                self.notify_message(f"Drawing file {path} empty")
                return

            columns_to_check = ["Time_in_s", "Frequency_in_Hz", "audiofilename"]
            found_columns = [ x in drawing.columns for x in columns_to_check ]
            if not all(found_columns):
                self.notify_message(f"Drawing file {path} malformed")
                return

            self.notify_message(f"Opening drawing file {path}")
            self.open_audio_file(drawing["audiofilename"][0])
            self.plot_spectrogram()
            self._input_data[self._filepointer].drawing = drawing
            self.plot_drawing()

    def func_annotate(self, b:bool):
        if len(self._input_data) == 0:
            return

        self.toggle_selector = RectangleSelector(
            self.canvas.axes,
            self.box_select_callback,
            # drawtype='box',
            useblit=False,
            button=[1],  # disable middle button
            interactive=False,
            # rectprops=dict(facecolor="blue", edgecolor="black", alpha=0.1, fill=True)
        )
        if self._input_data[self._filepointer].annotations is None:
            self._input_data[self._filepointer].annotations = pd.DataFrame(
                {
                    "t1": pd.Series(dtype="datetime64[ns]"),
                    "t2": pd.Series(dtype="datetime64[ns]"),
                    "f1": pd.Series(dtype="float"),
                    "f2": pd.Series(dtype="float"),
                    "label": pd.Series(dtype="object"),
                    "audiofilename": pd.Series(dtype="object"),
                }
            )
        for layout in [self.top2_layout, self.sidepanel_layout, self.plot_layout]:
            self.set_enabled_layout(layout, False)
        self.canvas.setEnabled(True)
        self.notify_message("Annotation mode")
        self.mpl_action_annotate = self.canvas.fig.canvas.mpl_connect("button_press_event", self.onclick_annotate)
        if hasattr(self, "mpl_action_draw"):
            self.canvas.fig.canvas.mpl_disconnect(self.mpl_action_draw)
        if not hasattr(self, "exitaction"):
            self.exitaction = QShortcut(QtCore.Qt.Key.Key_Return, self)
        else:
            self.exitaction.activated.disconnect()
        self.exitaction.activated.connect(self.func_annotate_save)

        if not hasattr(self, "abortaction"):
            self.abortaction = QShortcut(QtCore.Qt.Key.Key_Escape, self)
        else:
            self.abortaction.activated.disconnect()

        self.abortaction.activated.connect(self.func_annotate_abort)

    def display_all_artifacts(self):
        self.plot_spectrogram()
        self.plot_annotations()
        self.plot_drawing()

    def clear_spectrogram(self):
        self.plot_spectrogram()

    def func_playaudio(self):
        if not hasattr(self.canvas, "axes"):
            return

        if (lims := self.canvas.axes.get_xlim()):
            if int(lims[0]) == 0 and int(lims[1]) == 1:
                # no plot
                return

        if not hasattr(self, "play_obj"):
            new_rate = 32000

            t_limits = list(self.canvas.axes.get_xlim())

            t_limits = list(
                (
                    x
                    - self._input_data[self._filepointer].start
                    / self._input_data[self._filepointer].audio_data.fs
                    for x in t_limits
                )
            )
            f_limits = list(self.canvas.axes.get_ylim())
            if f_limits[1] >= (self._input_data[self._filepointer].audio_data.fs / 2):
                f_limits[1] = self._input_data[self._filepointer].audio_data.fs / 2 - 10

            x_select = self._input_data[self._filepointer].audio_data.x[
                int(t_limits[0] * self._input_data[self._filepointer].audio_data.fs) : int(
                    t_limits[1] * self._input_data[self._filepointer].audio_data.fs
                )
            ]

            sos = signal.butter(
                8,
                f_limits,
                "bandpass",
                fs=self._input_data[self._filepointer].audio_data.fs,
                output="sos",
            )
            x_select = signal.sosfilt(sos, x_select)

            number_of_samples = round(
                len(x_select)
                * (float(new_rate) / float(self.playbackspeed.currentText()))
                / self._input_data[self._filepointer].audio_data.fs
            )
            x_resampled = np.array(
                signal.resample(x_select, number_of_samples)
            ).astype("int")

            # normalize sound level
            maximum_x = 32767 * 0.8
            old_max = np.max(np.abs([x_resampled.min(), x_resampled.max()]))
            x_resampled = x_resampled * (maximum_x / old_max)
            x_resampled = x_resampled.astype(np.int16)

            wave_obj = sa.WaveObject(x_resampled, 1, 2, new_rate)

            self.play_obj = wave_obj.play()
        else:
            if self.play_obj.is_playing():
                sa.stop_all()
            else:
                if (lims := self.canvas.axes.get_xlim()):
                    if int(lims[0]) == 0 and int(lims[1]) == 1:
                        # no plot
                        return
                new_rate = 32000
                t_limits = list(self.canvas.axes.get_xlim())
                t_limits = list(
                    x
                    - self._input_data[self._filepointer].start
                    / self._input_data[self._filepointer].audio_data.fs
                    for x in t_limits
                )
                f_limits = list(self.canvas.axes.get_ylim())
                if f_limits[1] >= (self._input_data[self._filepointer].audio_data.fs / 2):
                    f_limits[1] = (
                        self._input_data[self._filepointer].audio_data.fs / 2 - 10
                    )

                x_select = self._input_data[self._filepointer].audio_data.x[
                    int(
                        t_limits[0] * self._input_data[self._filepointer].audio_data.fs
                    ) : int(
                        t_limits[1] * self._input_data[self._filepointer].audio_data.fs
                    )
                ]
                sos = signal.butter(
                    8,
                    f_limits,
                    "bandpass",
                    fs=self._input_data[self._filepointer].audio_data.fs,
                    output="sos",
                )
                x_select = signal.sosfilt(sos, x_select)

                # number_of_samples = round(len(x_select) * float(new_rate) / self._input_data[self._filepointer].audio_data.fs)
                number_of_samples = round(
                    len(x_select)
                    * (float(new_rate) / float(self.playbackspeed.currentText()))
                    / self._input_data[self._filepointer].audio_data.fs
                )

                x_resampled = np.array(
                    signal.resample(x_select, number_of_samples)
                ).astype("int")
                # normalize sound level
                maximum_x = 32767 * 0.8
                old_max = np.max(np.abs([x_resampled.min(), x_resampled.max()]))
                x_resampled = x_resampled * (maximum_x / old_max)
                x_resampled = x_resampled.astype(np.int16)

                wave_obj = sa.WaveObject(x_resampled, 1, 2, new_rate)
                self.play_obj = wave_obj.play()

    def func_saveaudio(self):
        if self._filepointer >= 0:
            savename = QtWidgets.QFileDialog.getSaveFileName(
                self,
                "QFileDialog.getSaveFileName()",
                self._output_directory,
                filter="wav files (*.wav)",
            )
            if len(savename[0]) > 0:
                savename = savename[0]
                new_rate = 32000

                t_limits = self.canvas.axes.get_xlim()
                f_limits = list(self.canvas.axes.get_ylim())
                if f_limits[1] >= (self._input_data[self._filepointer].audio_data.fs / 2):
                    f_limits[1] = self._input_data[self._filepointer].audio_data.fs / 2 - 10
                x_select = self._input_data[self._filepointer].audio_data.x[
                    int(t_limits[0] * self._input_data[self._filepointer].audio_data.fs) : int(
                        t_limits[1] * self._input_data[self._filepointer].audio_data.fs
                    )
                ]

                try:
                    sos = signal.butter(
                        8,
                        f_limits,
                        "bandpass",
                        fs=self._input_data[self._filepointer].audio_data.fs,
                        output="sos",
                    )
                except Exception as e:
                    self.notify_message(f"Could not save as wav file: {e}")
                else:
                    x_select = signal.sosfilt(sos, x_select)

                    number_of_samples = round(
                        len(x_select)
                        * (float(new_rate) / float(self.playbackspeed.currentText()))
                        / self._input_data[self._filepointer].audio_data.fs
                    )
                    x_resampled = np.array(
                        signal.resample(x_select, number_of_samples)
                    ).astype("int")
                    # normalize sound level
                    maximum_x = 32767 * 0.8
                    old_max = np.max(np.abs([x_resampled.min(), x_resampled.max()]))
                    x_resampled = x_resampled * (maximum_x / old_max)
                    x_resampled = x_resampled.astype(np.int16)

                    if savename[-4:] != ".wav":
                        savename = savename + ".wav"
                    wav.write(savename, new_rate, x_resampled)
        # button_save_audio.clicked.connect(func_saveaudio)

        # button_save_video=QtWidgets.QPushButton('Export video')

    def func_save_video(self):
        if self._filepointer >= 0:
            savename = QtWidgets.QFileDialog.getSaveFileName(
                self,
                "QFileDialog.getSaveFileName()",
                self._output_directory,
                filter="video files (*.mp4)",
            )
            if len(savename[0]) > 0:
                savename = savename[0]
                new_rate = 32000

                t_limits = self.canvas.axes.get_xlim()
                f_limits = list(self.canvas.axes.get_ylim())
                if f_limits[1] >= (self._input_data[self._filepointer].audio_data.fs / 2):
                    f_limits[1] = self._input_data[self._filepointer].audio_data.fs / 2 - 10
                x_select = self._input_data[self._filepointer].audio_data.x[
                    int(t_limits[0] * self._input_data[self._filepointer].audio_data.fs) : int(
                        t_limits[1] * self._input_data[self._filepointer].audio_data.fs
                    )
                ]
                try:
                    sos = signal.butter(
                        8,
                        f_limits,
                        "bandpass",
                        fs=self._input_data[self._filepointer].audio_data.fs,
                        output="sos",
                    )
                except Exception as e:
                    self.notify_message(f"Could not save as wav file: {e}")
                else:
                    x_select = signal.sosfilt(sos, x_select)

                    number_of_samples = round(
                        len(x_select)
                        * (float(new_rate) / float(self.playbackspeed.currentText()))
                        / self._input_data[self._filepointer].audio_data.fs
                    )
                    x_resampled = np.array(
                        signal.resample(x_select, number_of_samples)
                    ).astype("int")
                    # normalize sound level
                    maximum_x = 32767 * 0.8
                    old_max = np.max(np.abs([x_resampled.min(), x_resampled.max()]))
                    x_resampled = x_resampled * (maximum_x / old_max)
                    x_resampled = x_resampled.astype(np.int16)

                    if savename[:-4] == ".wav":
                        savename = savename[:-4]
                    if savename[:-4] == ".mp4":
                        savename = savename[:-4]
                    wav.write(savename + ".wav", new_rate, x_resampled)

                    # self.f_limits=self.canvas.axes.get_ylim()
                    # self.t_limits=self.canvas.axes.get_xlim()

                    audioclip = AudioFileClip(savename + ".wav")
                    duration = audioclip.duration
                    # plot_drawing()

                    self.canvas.axes.set_title(None)
                    # self.canvas.axes.set_ylim(f_limits)
                    # self.canvas.axes.set_xlim(t_limits)
                    self.line_2 = self.canvas.axes.plot(
                        [t_limits[0], t_limits[0]], f_limits, "-k"
                    )

                    def make_frame(x):
                        s = t_limits[1] - t_limits[0]
                        xx = x / duration * s + t_limits[0]
                        line = self.line_2.pop(0)
                        line.remove()
                        self.line_2 = self.canvas.axes.plot([xx, xx], f_limits, "-k")

                        return mplfig_to_npimage(self.canvas.fig)

                    animation = VideoClip(make_frame, duration=duration)
                    animation = animation.with_audio(audioclip)
                    animation.write_videofile(savename + ".mp4", fps=24, preset="fast")

