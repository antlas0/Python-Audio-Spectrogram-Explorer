# -*- coding: utf-8 -*-
"""
Created on Mon Sep  6 17:28:37 2021

@author: Sebastian Menze, sebastian.menze@gmail.com
@author: antlas0
"""

import sys
from PyQt6 import QtWidgets

from .mainwindow import MainWindow

def run():
    app = QtWidgets.QApplication(sys.argv)
    app.setApplicationName("Python Audio Spectrogram Explorer")
    w = MainWindow()
    sys.exit(app.exec())
