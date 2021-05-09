from PyQt5.QtCore import (pyqtSignal, QPoint, Qt, QRect, QSize)
from PyQt5.QtGui import QPainter,QColor, QTransform
from PyQt5.QtWidgets import (QLabel,QSizePolicy)
import cv2

class Image(QLabel):

    imageUpdated = pyqtSignal()

    def __init__(self, parent, scrollArea):
        QLabel.__init__(self, parent)
        #use scroll if image big
        self.vScrollbar = scrollArea.verticalScrollBar()
        self.hScrollbar = scrollArea.horizontalScrollBar()
        self.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.setMouseTracking(True)
        self.mouse_pressed = False
        self.crop_mode = False

    def set_imagePixmap(self, pixmap):
        self.img = pixmap   #save pixmap
        self.show()         #show

    def show(self):
        pm = self.img.copy()
        self.setPixmap(pm)
        self.imageUpdated.emit()


    def convert2arrayTemp(self):
        #current image convert to array
        temp=self.img.copy()
        img=temp.toImage()
        img.save('Images/temp.png','png')
        image=cv2.imread('Images/temp.png')
        return image

    def cropMode(self, enable):
        #crop mode open
        if enable:
            self.crop_mode = True
            self.scaleW = self.pixmap().width() / self.img.width()
            self.scaleH = self.pixmap().height() / self.img.height()
            self.pm_tmp = self.pixmap().copy()
            self.topleft = QPoint(0, 0)
            self.btmright = QPoint(self.pixmap().width() - 1, self.pixmap().height() - 1)
            self.last_pt = QPoint(self.btmright)
            self.p1, self.p2 = QPoint(self.topleft), QPoint(self.btmright)
            self.drawCropBox() #crop box
        #crop mode close
        else:
            self.crop_mode = False
            del self.pm_tmp
            self.show()

    def mousePressEvent(self, ev):
        self.clk_pos = ev.pos()
        self.clk_global = ev.globalPos()
        self.v_scrollbar_pos = self.vScrollbar.value()
        self.h_scrollbar_pos = self.hScrollbar.value()
        self.mouse_pressed = True
        if not self.crop_mode: return
        self.p1, self.p2 = QPoint(self.topleft), QPoint(self.btmright)  # Save position of cropbox
        # position clicked
        if QRect(self.topleft, QSize(60, 60)).contains(self.clk_pos):  #corner topleft is clicked (60x60)
            self.clk_area = 1
        elif QRect(self.btmright, QSize(-60, -60)).contains(self.clk_pos):  #corner bottom right is clicked (60x60)
            self.clk_area = 2
        elif QRect(self.topleft, self.btmright).contains(self.clk_pos):  # rectangle inside clicked (to move)
            self.clk_area = 3
        else:
            self.clk_area = 0

    def mouseReleaseEvent(self, ev):
        self.mouse_pressed = False
        if not self.crop_mode: return
        self.topleft, self.btmright = self.p1, self.p2

    def mouseMoveEvent(self, ev):
        if not self.mouse_pressed: return
        if not self.crop_mode:
            # Handle click and drag to scroll
            self.vScrollbar.setValue(self.v_scrollbar_pos + self.clk_global.y() - ev.globalY())
            self.hScrollbar.setValue(self.h_scrollbar_pos + self.clk_global.x() - ev.globalX())
            return

        moved = ev.pos() - self.clk_pos

        if self.clk_area == 1:  # Top left corner is clicked
            new_p1 = self.topleft + moved
            self.p1 = QPoint(max(0, new_p1.x()), max(0, new_p1.y()))

        elif self.clk_area == 2:  # Bottom right corner is clicked
            new_p2 = self.btmright + moved
            self.p2 = QPoint(min(self.last_pt.x(), new_p2.x()), min(self.last_pt.y(), new_p2.y()))

        elif self.clk_area == 3:  # clicked inside cropbox but none of the corner selected.
            min_dx, max_dx = -self.topleft.x(), self.last_pt.x() - self.btmright.x()
            min_dy, max_dy = -self.topleft.y(), self.last_pt.y() - self.btmright.y()
            dx = max(moved.x(), min_dx) if (moved.x() < 0) else min(moved.x(), max_dx)
            dy = max(moved.y(), min_dy) if (moved.y() < 0) else min(moved.y(), max_dy)
            self.p1 = self.topleft + QPoint(dx, dy)
            self.p2 = self.btmright + QPoint(dx, dy)

        self.drawCropBox()

    def drawCropBox(self):
        pm = self.pm_tmp.copy()
        pm_box = pm.copy(self.p1.x(), self.p1.y(), self.p2.x() - self.p1.x(), self.p2.y() - self.p1.y())
        painter = QPainter(pm)
        painter.fillRect(0, 0, pm.width(), pm.height(), QColor(127, 127, 127, 127))
        #Corners paint 20x20
        painter.drawPixmap(self.p1.x(), self.p1.y(), pm_box)
        painter.drawRect(self.p1.x(), self.p1.y(), self.p2.x() - self.p1.x(), self.p2.y() - self.p1.y())
        painter.drawRect(self.p1.x(), self.p1.y(), 20, 20)
        painter.drawRect(self.p2.x(), self.p2.y(), -20, -20)
        painter.setPen(Qt.white)
        painter.drawRect(self.p1.x() + 1, self.p1.y() + 1, 18, 18)
        painter.drawRect(self.p2.x() - 1, self.p2.y() - 1, -18, -18)
        painter.end()
        self.setPixmap(pm)
        self.imageUpdated.emit()

    def cropImage(self):
        #crop now button connection
        w, h = round((self.btmright.x() - self.topleft.x() + 1) / self.scaleW), round((self.btmright.y() - self.topleft.y() + 1) / self.scaleH)
        pm = self.img.copy(round(self.topleft.x() / self.scaleW), round(self.topleft.y() / self.scaleH), w, h)
        self.set_imagePixmap(pm)