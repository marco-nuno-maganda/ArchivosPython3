#!/usr/bin/python

"""
ZetCode PyQt5 tutorial

This example shows three labels on a window
using absolute positioning.

Author: Jan Bodnar
Website: zetcode.com
"""

import sys
from PyQt5.QtWidgets import QWidget, QLabel, QApplication
from PyQt5.QtCore import *


class Example(QWidget):

    def __init__(self):
        super().__init__()

        self.initUI()

    def initUI(self):
        lbl1 = QLabel('ZetCode', self)
        lbl1.resize(200,100)
        lbl1.setStyleSheet("background-color: red")
        lbl1.setAlignment(Qt.AlignCenter)
        lbl1.move(1, 1)

        lbl2 = QLabel('tutorials', self)
        lbl2.resize(200,100)
        lbl2.setAlignment(Qt.AlignCenter)
        lbl2.setStyleSheet("background-color: blue")
        lbl2.move(1, 101)

        lbl3 = QLabel('for programmers', self)
        lbl3.resize(200,100)
        lbl3.setAlignment(Qt.AlignCenter)
        lbl3.setStyleSheet("border: 4px solid black; background-color: green")
        lbl3.move(201, 1)

        lbl4 = QLabel('for programmers', self)
        lbl4.resize(200,100)
        lbl4.setAlignment(Qt.AlignCenter)
        lbl4.setStyleSheet("border: 4px solid black; background-color: yellow")
        lbl4.move(201, 100)

        # Col, Fila, Ancho, Alto
        #self.setGeometry(300, 300, 250, 150)
        self.setGeometry(300, 300, 640, 480)
        self.setWindowTitle('Absolute')
        self.show()


def main():
    app = QApplication(sys.argv)
    ex = Example()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
