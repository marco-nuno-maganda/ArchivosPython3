#!/usr/bin/python

"""
ZetCode PyQt5 tutorial

In this example, we position two push
buttons in the bottom-right corner
of the window.

Author: Jan Bodnar
Website: zetcode.com
"""

import sys
from PyQt5.QtWidgets import (QWidget, QPushButton, QLabel, QLineEdit,
                             QHBoxLayout, QVBoxLayout, QApplication)
from PyQt5.QtCore import *

class Example(QWidget):

    def __init__(self):
        super().__init__()

        self.initUI()
        
    def clicked (self):
        X = self.textbox.text()
        print ("Presionaste boton OK, contenido de Textbox",X)      

    def OnePiece (self):
        print ("Presionaste boton CANCEL")      


    def initUI(self):

        okButton = QPushButton("OK")
        okButton.clicked.connect(self.clicked)
        
        cancelButton = QPushButton("Cancel")
        cancelButton.clicked.connect(self.OnePiece)
        
        self.textbox = QLineEdit()
        
        hbox = QHBoxLayout()
        #hbox.addStretch(1)
        #hbox.addWidget(cancelButton)
        #hbox.addWidget(okButton)
        
        hbox2 = QHBoxLayout()
        
        #lblA = QLabel('Lenguajes y Automatas')
        #lblA.setAlignment(Qt.AlignCenter)
        #lblA.setStyleSheet("QLabel {background-color:}")
        #hbox2.addWidget(lblA)

        #vbox = QVBoxLayout()
        #vbox.addLayout(hbox)
        #vbox.addLayout(hbox2)
        #vbox.addStretch(1)
        
        vbox2 = QVBoxLayout()
        vbox2.addWidget(cancelButton)
        vbox2.addWidget(self.textbox)
        vbox2.addWidget(okButton)
        lblA = QLabel('Lenguajes y Automatas')
        lblA.setStyleSheet("QLabel {background-color: red;}")
        lblA.setAlignment(Qt.AlignCenter)
        
        hbox2.addLayout(vbox2)
        hbox2.addWidget(lblA)
        hbox2.addStretch(1)

        #self.setLayout(vbox)
        self.setLayout(hbox2)

        self.setGeometry(300, 300, 640, 480)
        self.setWindowTitle('Buttons')
        self.show()


def main():
    app = QApplication(sys.argv)
    ex = Example()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
