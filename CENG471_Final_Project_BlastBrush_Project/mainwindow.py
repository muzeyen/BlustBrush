# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'mainwindow.ui'
#
# Created by: PyQt5 UI code generator 5.15.2
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        iconn = QtGui.QIcon()
        iconn.addPixmap(QtGui.QPixmap("Icons/blastbrush.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        MainWindow.setWindowIcon(iconn)
        MainWindow.resize(975, 710)
        MainWindow.setMinimumSize(QtCore.QSize(613, 409))
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.horizontalLayout = QtWidgets.QHBoxLayout(self.centralwidget)
        self.horizontalLayout.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout.setSpacing(0)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.frame = QtWidgets.QFrame(self.centralwidget)
        self.frame.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame.setObjectName("frame")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.frame)
        self.verticalLayout.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout.setObjectName("verticalLayout")
        self.openBtn = QtWidgets.QPushButton(self.frame)
        self.openBtn.setText("")
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap("Icons/import.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.openBtn.setIcon(icon)
        self.openBtn.setIconSize(QtCore.QSize(32, 32))
        self.openBtn.setObjectName("openBtn")
        self.verticalLayout.addWidget(self.openBtn)
        self.saveBtn = QtWidgets.QPushButton(self.frame)
        self.saveBtn.setText("")
        icon1 = QtGui.QIcon()
        icon1.addPixmap(QtGui.QPixmap("Icons/save.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.saveBtn.setIcon(icon1)
        self.saveBtn.setIconSize(QtCore.QSize(32, 32))
        self.saveBtn.setObjectName("saveBtn")
        self.verticalLayout.addWidget(self.saveBtn)
        self.cropBtn = QtWidgets.QPushButton(self.frame)
        self.cropBtn.setText("")
        icon2 = QtGui.QIcon()
        icon2.addPixmap(QtGui.QPixmap("Icons/crop.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.cropBtn.setIcon(icon2)
        self.cropBtn.setIconSize(QtCore.QSize(32, 32))
        self.cropBtn.setObjectName("cropBtn")
        self.verticalLayout.addWidget(self.cropBtn)
        self.enhancementBtn = QtWidgets.QPushButton(self.frame)
        font = QtGui.QFont()
        font.setPointSize(8)
        self.enhancementBtn.setFont(font)
        self.enhancementBtn.setObjectName("enhancementBtn")
        self.verticalLayout.addWidget(self.enhancementBtn)
        self.label = QtWidgets.QLabel(self.frame)
        font = QtGui.QFont()
        font.setPointSize(10)
        self.label.setFont(font)
        self.label.setAlignment(QtCore.Qt.AlignCenter)
        self.label.setObjectName("label")
        self.verticalLayout.addWidget(self.label)
        self.horizontalWidget = QtWidgets.QWidget(self.frame)
        self.horizontalWidget.setMinimumSize(QtCore.QSize(50, 50))
        self.horizontalWidget.setObjectName("horizontalWidget")
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout(self.horizontalWidget)
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.verticalFlipBtn = QtWidgets.QPushButton(self.horizontalWidget)
        self.verticalFlipBtn.setText("")
        icon4 = QtGui.QIcon()
        icon4.addPixmap(QtGui.QPixmap("Icons/vertical.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.verticalFlipBtn.setIcon(icon4)
        self.verticalFlipBtn.setIconSize(QtCore.QSize(32, 32))
        self.verticalFlipBtn.setObjectName("verticalFlipBtn")
        self.horizontalLayout_2.addWidget(self.verticalFlipBtn)
        self.horizontalFlipBtn = QtWidgets.QPushButton(self.horizontalWidget)
        self.horizontalFlipBtn.setText("")
        icon5 = QtGui.QIcon()
        icon5.addPixmap(QtGui.QPixmap("Icons/horizontal.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.horizontalFlipBtn.setIcon(icon5)
        self.horizontalFlipBtn.setIconSize(QtCore.QSize(32, 32))
        self.horizontalFlipBtn.setObjectName("horizontalFlipBtn")
        self.horizontalLayout_2.addWidget(self.horizontalFlipBtn)
        self.verticalLayout.addWidget(self.horizontalWidget)
        self.label_2 = QtWidgets.QLabel(self.frame)
        font = QtGui.QFont()
        font.setPointSize(10)
        self.label_2.setFont(font)
        self.label_2.setAlignment(QtCore.Qt.AlignCenter)
        self.label_2.setObjectName("label_2")
        self.verticalLayout.addWidget(self.label_2)
        self.horizontalWidget1 = QtWidgets.QWidget(self.frame)
        self.horizontalWidget1.setMinimumSize(QtCore.QSize(50, 20))
        self.horizontalWidget1.setObjectName("horizontalWidget1")
        self.horizontalLayout_3 = QtWidgets.QHBoxLayout(self.horizontalWidget1)
        self.horizontalLayout_3.setSpacing(8)
        self.horizontalLayout_3.setObjectName("horizontalLayout_3")
        self.leftRotateBtn = QtWidgets.QPushButton(self.horizontalWidget1)
        self.leftRotateBtn.setText("")
        icon6 = QtGui.QIcon()
        icon6.addPixmap(QtGui.QPixmap("Icons/leftRotate.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.leftRotateBtn.setIcon(icon6)
        self.leftRotateBtn.setIconSize(QtCore.QSize(32, 32))
        self.leftRotateBtn.setObjectName("leftRotateBtn")
        self.horizontalLayout_3.addWidget(self.leftRotateBtn)
        self.rightRotateBtn = QtWidgets.QPushButton(self.horizontalWidget1)
        self.rightRotateBtn.setText("")
        icon7 = QtGui.QIcon()
        icon7.addPixmap(QtGui.QPixmap("Icons/r.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.rightRotateBtn.setIcon(icon7)
        self.rightRotateBtn.setIconSize(QtCore.QSize(32, 32))
        self.rightRotateBtn.setObjectName("rightRotateBtn")
        self.horizontalLayout_3.addWidget(self.rightRotateBtn)
        self.verticalLayout.addWidget(self.horizontalWidget1)
        self.label_3 = QtWidgets.QLabel(self.frame)
        font = QtGui.QFont()
        font.setPointSize(10)
        self.label_3.setFont(font)
        self.label_3.setAlignment(QtCore.Qt.AlignCenter)
        self.label_3.setObjectName("label_3")
        self.verticalLayout.addWidget(self.label_3)
        self.horizontalWidget2 = QtWidgets.QWidget(self.frame)
        self.horizontalWidget2.setMinimumSize(QtCore.QSize(50, 50))
        self.horizontalWidget2.setObjectName("horizontalWidget2")
        self.horizontalLayout_4 = QtWidgets.QHBoxLayout(self.horizontalWidget2)
        self.horizontalLayout_4.setObjectName("horizontalLayout_4")
        self.brightnessUpBtn = QtWidgets.QPushButton(self.horizontalWidget2)
        self.brightnessUpBtn.setText("")
        icon8 = QtGui.QIcon()
        icon8.addPixmap(QtGui.QPixmap("Icons/plus.webp"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.brightnessUpBtn.setIcon(icon8)
        self.brightnessUpBtn.setIconSize(QtCore.QSize(32, 32))
        self.brightnessUpBtn.setObjectName("brightnessUpBtn")
        self.horizontalLayout_4.addWidget(self.brightnessUpBtn)
        self.brightnessDownBtn = QtWidgets.QPushButton(self.horizontalWidget2)
        self.brightnessDownBtn.setText("")
        icon9 = QtGui.QIcon()
        icon9.addPixmap(QtGui.QPixmap("Icons/minus.ico"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.brightnessDownBtn.setIcon(icon9)
        self.brightnessDownBtn.setIconSize(QtCore.QSize(32, 32))
        self.brightnessDownBtn.setObjectName("brightnessDownBtn")
        self.horizontalLayout_4.addWidget(self.brightnessDownBtn)
        self.verticalLayout.addWidget(self.horizontalWidget2)
        self.label_4 = QtWidgets.QLabel(self.frame)
        font = QtGui.QFont()
        font.setPointSize(10)
        self.label_4.setFont(font)
        self.label_4.setAlignment(QtCore.Qt.AlignCenter)
        self.label_4.setObjectName("label_4")
        self.verticalLayout.addWidget(self.label_4)
        self.horizontalWidget3 = QtWidgets.QWidget(self.frame)
        self.horizontalWidget3.setMinimumSize(QtCore.QSize(50, 50))
        self.horizontalWidget3.setObjectName("horizontalWidget3")
        self.horizontalLayout_5 = QtWidgets.QHBoxLayout(self.horizontalWidget3)
        self.horizontalLayout_5.setObjectName("horizontalLayout_5")
        self.contrastUpBtn = QtWidgets.QPushButton(self.horizontalWidget3)
        self.contrastUpBtn.setText("")
        self.contrastUpBtn.setIcon(icon8)
        self.contrastUpBtn.setIconSize(QtCore.QSize(32, 32))
        self.contrastUpBtn.setObjectName("contrastUpBtn")
        self.horizontalLayout_5.addWidget(self.contrastUpBtn)
        self.contrastDownBtn = QtWidgets.QPushButton(self.horizontalWidget3)
        self.contrastDownBtn.setText("")
        self.contrastDownBtn.setIcon(icon9)
        self.contrastDownBtn.setIconSize(QtCore.QSize(32, 32))
        self.contrastDownBtn.setObjectName("contrastDownBtn")
        self.horizontalLayout_5.addWidget(self.contrastDownBtn)
        self.verticalLayout.addWidget(self.horizontalWidget3)
        spacerItem = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.verticalLayout.addItem(spacerItem)
        self.quitBtn = QtWidgets.QPushButton(self.frame)
        self.quitBtn.setText("")
        icon10 = QtGui.QIcon()
        icon10.addPixmap(QtGui.QPixmap("Icons/exit.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.quitBtn.setIcon(icon10)
        self.quitBtn.setIconSize(QtCore.QSize(32, 32))
        self.quitBtn.setObjectName("quitBtn")
        self.verticalLayout.addWidget(self.quitBtn)
        self.horizontalLayout.addWidget(self.frame)
        self.scrollArea = QtWidgets.QScrollArea(self.centralwidget)
        self.scrollArea.setWidgetResizable(True)
        self.scrollArea.setObjectName("scrollArea")
        self.scrollAreaWidgetContents = QtWidgets.QWidget()
        self.scrollAreaWidgetContents.setGeometry(QtCore.QRect(0, 0, 744, 683))
        self.scrollAreaWidgetContents.setObjectName("scrollAreaWidgetContents")
        self.scrollArea.setWidget(self.scrollAreaWidgetContents)
        self.horizontalLayout.addWidget(self.scrollArea)
        self.frame_2 = QtWidgets.QFrame(self.centralwidget)
        self.frame_2.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_2.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_2.setObjectName("frame_2")
        self.verticalLayout_2 = QtWidgets.QVBoxLayout(self.frame_2)
        self.verticalLayout_2.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.effect1 = QtWidgets.QPushButton(self.frame_2)
        self.effect1.setObjectName("effect1")
        self.verticalLayout_2.addWidget(self.effect1)
        self.effect2 = QtWidgets.QPushButton(self.frame_2)
        self.effect2.setObjectName("effect2")
        self.verticalLayout_2.addWidget(self.effect2)
        self.effect3 = QtWidgets.QPushButton(self.frame_2)
        self.effect3.setIconSize(QtCore.QSize(32, 32))
        self.effect3.setObjectName("effect3")
        self.verticalLayout_2.addWidget(self.effect3)
        self.effect4 = QtWidgets.QPushButton(self.frame_2)
        self.effect4.setObjectName("effect4")
        self.verticalLayout_2.addWidget(self.effect4)
        self.effect5 = QtWidgets.QPushButton(self.frame_2)
        self.effect5.setObjectName("effect5")
        self.verticalLayout_2.addWidget(self.effect5)
        self.effect6 = QtWidgets.QPushButton(self.frame_2)
        self.effect6.setObjectName("effect6")
        self.verticalLayout_2.addWidget(self.effect6)
        self.effect7 = QtWidgets.QPushButton(self.frame_2)
        self.effect7.setObjectName("effect7")
        self.verticalLayout_2.addWidget(self.effect7)
        self.effect8 = QtWidgets.QPushButton(self.frame_2)
        self.effect8.setObjectName("effect8")
        self.verticalLayout_2.addWidget(self.effect8)
        self.effect9 = QtWidgets.QPushButton(self.frame_2)
        self.effect9.setObjectName("effect9")
        self.verticalLayout_2.addWidget(self.effect9)
        self.effect10 = QtWidgets.QPushButton(self.frame_2)
        self.effect10.setObjectName("effect10")
        self.verticalLayout_2.addWidget(self.effect10)
        self.effect11 = QtWidgets.QPushButton(self.frame_2)
        self.effect11.setObjectName("effect11")
        self.verticalLayout_2.addWidget(self.effect11)
        self.effect12 = QtWidgets.QPushButton(self.frame_2)
        self.effect12.setObjectName("effect12")
        self.verticalLayout_2.addWidget(self.effect12)
        self.effect13 = QtWidgets.QPushButton(self.frame_2)
        self.effect13.setObjectName("effect13")
        self.verticalLayout_2.addWidget(self.effect13)
        self.effect14 = QtWidgets.QPushButton(self.frame_2)
        self.effect14.setObjectName("effect14")
        self.verticalLayout_2.addWidget(self.effect14)
        self.effect15 = QtWidgets.QPushButton(self.frame_2)
        self.effect15.setObjectName("effect15")
        self.verticalLayout_2.addWidget(self.effect15)
        self.effect16 = QtWidgets.QPushButton(self.frame_2)
        self.effect16.setObjectName("effect16")
        self.verticalLayout_2.addWidget(self.effect16)
        self.effect17 = QtWidgets.QPushButton(self.frame_2)
        self.effect17.setObjectName("effect17")
        self.verticalLayout_2.addWidget(self.effect17)
        self.effect18 = QtWidgets.QPushButton(self.frame_2)
        icon16 = QtGui.QIcon()
        self.effect18.setObjectName("effect18")
        self.verticalLayout_2.addWidget(self.effect18)
        self.effect19 = QtWidgets.QPushButton(self.frame_2)
        self.effect19.setIconSize(QtCore.QSize(32, 32))
        self.effect19.setObjectName("effect19")
        self.verticalLayout_2.addWidget(self.effect19)
        self.effect20 = QtWidgets.QPushButton(self.frame_2)
        self.effect20.setCheckable(True)
        self.effect20.setObjectName("effect20")
        self.verticalLayout_2.addWidget(self.effect20)
        self.horizontalLayout.addWidget(self.frame_2)
        MainWindow.setCentralWidget(self.centralwidget)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "BlastBrush"))
        self.enhancementBtn.setText(_translate("MainWindow", "ENHANCEMENT"))
        self.label.setText(_translate("MainWindow", "FLIP"))
        self.label_2.setText(_translate("MainWindow", "ROTATE"))
        self.label_3.setText(_translate("MainWindow", "BRIGHTNESS"))
        self.label_4.setText(_translate("MainWindow", "CONTRAST"))
        self.effect1.setText(_translate("MainWindow", "BLUE NIGHT"))
        self.effect2.setText(_translate("MainWindow", "PUMPKIN SEASON"))
        self.effect3.setText(_translate("MainWindow", "COTTON CANDY"))
        self.effect4.setText(_translate("MainWindow", "FOREST FIRE"))
        self.effect5.setText(_translate("MainWindow", "BRUSH MARK"))
        self.effect6.setText(_translate("MainWindow", "SKETCH"))
        self.effect7.setText(_translate("MainWindow", "THE MOON"))
        self.effect8.setText(_translate("MainWindow", "METAL SHINE"))
        self.effect9.setText(_translate("MainWindow", "CARTOON"))
        self.effect10.setText(_translate("MainWindow", "FIREWORK"))
        self.effect11.setText(_translate("MainWindow", "CALIFORNIA DAYS"))
        self.effect12.setText(_translate("MainWindow", "DEEP PURPLE"))
        self.effect13.setText(_translate("MainWindow", "OIL PAINT"))
        self.effect14.setText(_translate("MainWindow", "SPLASH"))
        self.effect15.setText(_translate("MainWindow", "MAGIC PENCIL"))
        self.effect16.setText(_translate("MainWindow", "SEPIA"))
        self.effect17.setText(_translate("MainWindow", "CHECKBOX BLACKMARK"))
        self.effect18.setText(_translate("MainWindow", "NOISE"))
        self.effect19.setText(_translate("MainWindow", "NEGATIVE"))
        self.effect20.setText(_translate("MainWindow", "CIRCLE MOSAIC"))



if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
