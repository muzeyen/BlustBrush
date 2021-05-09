import sys
from PyQt5.QtGui import QPixmap, QImageReader, QImage, qRgb
from PyQt5.QtWidgets import (QApplication, QMainWindow, QHBoxLayout,  QFileDialog, QPushButton)
from scipy.interpolate import UnivariateSpline
from mainwindow import Ui_MainWindow
import numpy as np
from image import *

gray_color_table = [qRgb(i, i, i) for i in range(256)]

class Window(QMainWindow, Ui_MainWindow):
    def __init__(self):
        QMainWindow.__init__(self)
        Ui_MainWindow.setupUi(self, self)
        layout = QHBoxLayout(self.scrollAreaWidgetContents)
        self.imageLabel = Image(self, self.scrollArea)
        layout.addWidget(self.imageLabel)
        self.connects()
        desktop = QApplication.desktop()
        self.screen_width = desktop.availableGeometry().width()
        self.screen_height = desktop.availableGeometry().height()
        self.filepath = ''
        self.crop_widgets = []

    def connects(self):
        #connection to functions
        self.openBtn.clicked.connect(self.openFile)
        self.saveBtn.clicked.connect(self.saveFile)
        self.cropBtn.clicked.connect(self.cropImage)
        self.enhancementBtn.clicked.connect(self.enhancement)
        self.verticalFlipBtn.clicked.connect(self.verticalFlip)
        self.horizontalFlipBtn.clicked.connect(self.horizontalFlip)
        self.rightRotateBtn.clicked.connect(self.rotateRight)
        self.leftRotateBtn.clicked.connect(self.rotateLeft)
        self.brightnessDownBtn.clicked.connect(self.brightnessDown)
        self.brightnessUpBtn.clicked.connect(self.brightnessUp)
        self.contrastDownBtn.clicked.connect(self.contrastDown)
        self.contrastUpBtn.clicked.connect(self.contrastUp)
        self.quitBtn.clicked.connect(self.close)
        # Effects
        self.effect1.clicked.connect(self.__effect1)
        self.effect2.clicked.connect(self.__effect2)
        self.effect3.clicked.connect(self.__effect3)
        self.effect4.clicked.connect(self.__effect4)
        self.effect5.clicked.connect(self.__effect5)
        self.effect6.clicked.connect(self.__effect6)
        self.effect7.clicked.connect(self.__effect7)
        self.effect8.clicked.connect(self.__effect8)
        self.effect9.clicked.connect(self.__effect9)
        self.effect10.clicked.connect(self.__effect10)
        self.effect11.clicked.connect(self.__effect11)
        self.effect12.clicked.connect(self.__effect12)
        self.effect13.clicked.connect(self.__effect13)
        self.effect14.clicked.connect(self.__effect14)
        self.effect15.clicked.connect(self.__effect15)
        self.effect16.clicked.connect(self.__effect16)
        self.effect17.clicked.connect(self.__effect17)
        self.effect18.clicked.connect(self.__effect18)
        self.effect19.clicked.connect(self.__effect19)
        self.effect20.clicked.connect(self.__effect20)

    def convert2arrayOriginal(self):
        #convert to array current image
        temp=self.pm
        img=temp.toImage()
        img.save('Images/originalTemp.png','png')
        image=cv2.imread('Images/originalTemp.png')
        return image

    def convertCvToQImage(self, img):
        #for the image shapes convert QImage to print label
        # if input image format is openCv image format np.uint8
        if img.dtype == np.uint8:
            # grayscale image or images having two dimensions [height, width]
            if len(img.shape) == 2:
                qim = QImage(img.data, img.shape[1], img.shape[0], img.strides[0], QImage.Format_Indexed8)
                qim.setColorTable(gray_color_table)
                return qim
            # image having three dimansions [height, width, nChannels]
            elif len(img.shape) == 3:
                # if image has three channels
                if img.shape[2] == 3:
                    qim = QImage(img.data, img.shape[1], img.shape[0], img.strides[0], QImage.Format_BGR888)
                    return qim
                # if image has four channels
                elif img.shape[2] == 4:
                    qim = QImage(img.data, img.shape[1], img.shape[0], img.strides[0], QImage.Format_ARGB32)
                    return qim

    def openFile(self):
        #open image from file
        filepath, _ = QFileDialog.getOpenFileName(self, 'Open Image', self.filepath, "JPG Images (*.jpg);;PNG Images (*.png)")
        if filepath == '':
            return
        print(filepath)
        image_reader = QImageReader(filepath)
        image_reader.setAutoTransform(True)
        self.pm = QPixmap.fromImageReader(image_reader)
        if not self.pm.isNull() :
            self.imageLabel.set_imagePixmap(self.pm)
        else:
            return
        self.filepath = filepath

    def saveFile(self):
        #save image
        filepath, sel_filter = QFileDialog.getSaveFileName(self, 'Save Image','Results\output.png',"JPEG Image (*.jpg);;PNG Image (*.png)")
        if filepath != '':
            pm = self.imageLabel.img
            if not pm.isNull():
                pm.save(filepath, None)

    def cropImage(self):
        #connection to crop button
        #crop mode true
        if not self.imageLabel.crop_mode:
            self.imageLabel.cropMode(True)
            #status bar enable
            cropnowBtn = QPushButton("Crop Now", self.statusbar)
            self.statusbar.addPermanentWidget(cropnowBtn)
            cropcancelBtn = QPushButton("Cancel", self.statusbar)
            self.statusbar.addPermanentWidget(cropcancelBtn)
            cropnowBtn.clicked.connect(self.imageLabel.cropImage)
            cropnowBtn.clicked.connect(self.cancelCropping)
            cropcancelBtn.clicked.connect(self.cancelCropping)
            self.crop_widgets += [cropnowBtn, cropcancelBtn]

    def cancelCropping(self):
        #status bar disable
        self.imageLabel.cropMode(False)
        while len(self.crop_widgets)> 0:
            widget = self.crop_widgets.pop()
            self.statusbar.removeWidget(widget)
            widget.deleteLater()

    def enhancement(self):
        image = self.imageLabel.convert2arrayTemp()
        # segregate color streams
        img_yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
        # apply histogram equalization
        img_yuv[:, :, 0] = cv2.equalizeHist(img_yuv[:, :, 0])
        hist_eq = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)
        qimage = self.convertCvToQImage(hist_eq)
        q = QPixmap(qimage)
        self.imageLabel.set_imagePixmap(q)


    def verticalFlip(self):
        image= self.imageLabel.convert2arrayTemp()
        h, w, _= image.shape
        temp = np.zeros((h, w, 3), np.uint8)
        for j in range(0, h):
            temp[j, :, :] = image[h - j - 1, :, :]
        image2 = temp
        qimage=QImage(image2, w,h,3*w, QImage.Format_BGR888)
        q=QPixmap(qimage)
        self.imageLabel.set_imagePixmap(q)

    def horizontalFlip(self):
        image = self.imageLabel.convert2arrayTemp()
        h, w, _ = image.shape
        temp = np.zeros((h, w, 3), np.uint8)
        for i in range(0, w):
            temp[:, i, :] = image[:, w - i - 1, :]
        image2 = temp
        qimage = QImage(image2, w, h, 3 * w, QImage.Format_BGR888)
        q = QPixmap(qimage)
        self.imageLabel.set_imagePixmap(q)

    def rotateLeft(self):
        image= self.imageLabel.img
        transform = QTransform()
        transform.rotate(270)
        image= image.transformed(transform)
        self.imageLabel.set_imagePixmap(image)

    def rotateRight(self):
        image = self.imageLabel.img
        transform = QTransform()
        transform.rotate(90)
        image = image.transformed(transform)
        self.imageLabel.set_imagePixmap(image)

    def brightnessDown(self):
        img=self.imageLabel.convert2arrayTemp()
        h, w, _ = img.shape
        image = cv2.addWeighted(img, 1, np.zeros(img.shape, img.dtype), 0, -10)
        qimage = QImage(image, w, h, 3 * w, QImage.Format_BGR888)
        q = QPixmap(qimage)
        self.imageLabel.set_imagePixmap(q)

    def brightnessUp(self):
        img = self.imageLabel.convert2arrayTemp()
        h, w, _ = img.shape
        image = cv2.addWeighted(img, 1, np.zeros(img.shape, img.dtype), 0,10)
        qimage = QImage(image, w, h, 3 * w, QImage.Format_BGR888)
        q = QPixmap(qimage)
        self.imageLabel.set_imagePixmap(q)

    def contrastDown(self):
        img = self.imageLabel.convert2arrayTemp()
        h, w, _ = img.shape
        image = cv2.addWeighted(img, 0.7, np.zeros(img.shape, img.dtype), 0, 0)
        qimage = QImage(image, w, h, 3 * w, QImage.Format_BGR888)
        q = QPixmap(qimage)
        self.imageLabel.set_imagePixmap(q)

    def contrastUp(self):
        img = self.imageLabel.convert2arrayTemp()
        h, w, _ = img.shape
        image = cv2.addWeighted(img, 1.25, np.zeros(img.shape, img.dtype), 0, 0)
        qimage = QImage(image, w, h, 3 * w, QImage.Format_BGR888)
        q = QPixmap(qimage)
        self.imageLabel.set_imagePixmap(q)

#------------------------------------#
    #EFFECTS#

    def __effect1(self):
        #BLUE NIGHT
        img = self.convert2arrayOriginal()
        h, w, _ = img.shape
        table_r = UnivariateSpline([0, 64, 128, 256], [0, 80, 160, 256])(range(256))
        table_b = UnivariateSpline([0, 64, 128, 256], [0, 50, 100, 256])(range(256))
        R, G, B = cv2.split(img)
        red_Val = cv2.LUT(R, table_r).astype(np.uint8)
        blue_Val = cv2.LUT(B, table_b).astype(np.uint8)
        image = cv2.merge((red_Val, G, blue_Val))
        qimage = self.convertCvToQImage(image)
        q = QPixmap(qimage)
        self.imageLabel.set_imagePixmap(q)

    def __effect2(self):
        #PUMPKIN SEASON
        img =self.convert2arrayOriginal()
        h, w, _ = img.shape
        table_b = UnivariateSpline([0, 64, 128, 256], [0, 80, 160, 256])(range(256))
        table_r = UnivariateSpline([0, 64, 128, 256], [0, 50, 100, 256])(range(256))
        R, G, B = cv2.split(img)
        red_Val = cv2.LUT(R, table_r).astype(np.uint8)
        blue_Val = cv2.LUT(B, table_b).astype(np.uint8)
        image = cv2.merge((red_Val, G, blue_Val))
        qimage = self.convertCvToQImage(image)
        q = QPixmap(qimage)
        self.imageLabel.set_imagePixmap(q)

    def __effect3(self):
        #COTTON CANDY
        img = self.convert2arrayOriginal()
        h, w, _ = img.shape
        table = UnivariateSpline([0, 64, 128, 256], [0, 80, 160, 256])(range(256))
        R, G, B = cv2.split(img)
        red_Val = cv2.LUT(R, table).astype(np.uint8)
        blue_Val = cv2.LUT(B, table).astype(np.uint8)
        image = cv2.merge((red_Val, G, blue_Val))
        qimage = self.convertCvToQImage(image)
        q = QPixmap(qimage)
        self.imageLabel.set_imagePixmap(q)

    def __effect4(self):
        #FOREST FIRE
        img = self.convert2arrayOriginal()
        h, w, _ = img.shape
        orig_Val = np.array([0, 50, 100, 150, 200, 255])
        red_Val = np.array([0, 20, 10, 50, 150, 255])
        blue_Val = np.array([0, 70, 100, 190, 200, 255])
        all_Val = np.arange(0, 256)
        r_table = np.interp(all_Val, orig_Val, red_Val)
        b_table = np.interp(all_Val, orig_Val, blue_Val)
        B, G, R = cv2.split(img)
        R = cv2.LUT(R, r_table).astype(np.uint8)
        B = cv2.LUT(B, b_table).astype(np.uint8)
        image = cv2.merge([B, G, R])
        qimage = self.convertCvToQImage(image)
        q = QPixmap(qimage)
        self.imageLabel.set_imagePixmap(q)

    def __effect5(self):
        #BRUSH MARK
        img = self.convert2arrayOriginal()
        h, w, _ = img.shape
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.medianBlur(gray, 5)
        sketch = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 9)
        color = cv2.bilateralFilter(img, 9, 300, 300)
        image = cv2.bitwise_and(color, color, mask=sketch)
        qimage = self.convertCvToQImage(image)
        q = QPixmap(qimage)
        self.imageLabel.set_imagePixmap(q)

    def __effect6(self):
        #SKETCH
        img = self.convert2arrayOriginal()
        h, w,_ = img.shape
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Blur the image using Gaussian Blur
        gray_blur = cv2.GaussianBlur(gray, (25, 25), 0)
        # Convert the image into pencil sketch
        image = cv2.divide(gray, gray_blur, scale=250.0)
        qimage = self.convertCvToQImage(image)
        q = QPixmap(qimage)
        self.imageLabel.set_imagePixmap(q)

    def __effect7(self):
        #THE MOON
        img = self.convert2arrayOriginal()
        h, w, _ = img.shape
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        qimage = self.convertCvToQImage(gray)
        q = QPixmap(qimage)
        self.imageLabel.set_imagePixmap(q)

    def __effect8(self):
        #METAL SHINE
        img = self.convert2arrayOriginal()
        h, w, _ = img.shape
        mask = np.array([[0, -1, -1], [1, 0, -1],[1, 1, 0]])
        image= cv2.filter2D(img, -1, mask)
        qimage = self.convertCvToQImage(image)
        q = QPixmap(qimage)
        self.imageLabel.set_imagePixmap(q)

    def __effect9(self):
        #CARTOON
        img = self.convert2arrayOriginal()
        kernel = np.array([[2, -6, -5], [3, 6, -8], [2, 1, 6]])
        new_image = cv2.filter2D(img, -1, kernel)
        qimage = self.convertCvToQImage(new_image)
        q = QPixmap(qimage)
        self.imageLabel.set_imagePixmap(q)

    def __effect10(self):
        #FIREWORK
        img = self.convert2arrayOriginal()
        img = cv2.applyColorMap(img, cv2.COLORMAP_SPRING)
        qimage = self.convertCvToQImage(img)
        q = QPixmap(qimage)
        self.imageLabel.set_imagePixmap(q)

    def __effect11(self):
        #CALIFORNIA DAYS
        img = self.convert2arrayOriginal()
        h, w, _ = img.shape
        table_b = UnivariateSpline([0, 64, 128, 256], [0, 80, 160, 256])(range(256))
        table_g = UnivariateSpline([0, 64, 128, 256], [0, 50, 100, 256])(range(256))
        R, G, B = cv2.split(img)
        blue_Val = cv2.LUT(B, table_b).astype(np.uint8)
        green_Val=cv2.LUT(B, table_g).astype(np.uint8)
        image = cv2.merge((R, green_Val, blue_Val))
        qimage = self.convertCvToQImage(image)
        q = QPixmap(qimage)
        self.imageLabel.set_imagePixmap(q)

    def __effect12(self):
        #DEEP PURPLE
        img = self.convert2arrayOriginal()
        h, w, _ = img.shape
        table_r = UnivariateSpline([0, 70, 180, 256], [0, 80, 180, 256])(range(256))
        table_g = UnivariateSpline([30, 70, 180, 256], [20, 50, 180, 256])(range(256))
        R, G, B = cv2.split(img)
        red_Val = cv2.LUT(B, table_r).astype(np.uint8)
        green_Val = cv2.LUT(G, table_g).astype(np.uint8)
        image = cv2.merge((red_Val, green_Val, B))
        qimage = self.convertCvToQImage(image)
        q = QPixmap(qimage)
        self.imageLabel.set_imagePixmap(q)

    def __effect13(self):
        #OIL PAINT
        img = self.convert2arrayOriginal()
        h, w, _ = img.shape
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (6, 6))
        morph = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
        # brighten dark regions
        result = cv2.normalize(morph, None, 20, 255, cv2.NORM_MINMAX)
        qimage = self.convertCvToQImage(result)
        q = QPixmap(qimage)
        self.imageLabel.set_imagePixmap(q)

    def hsv(self,img, l, u):
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        lower = np.array([l, 128, 128])  # setting lower HSV value
        upper = np.array([u, 255, 255])  # setting upper HSV value
        mask = cv2.inRange(hsv, lower, upper)  # generating mask
        return mask

    def __effect14(self):
        #SPLASH
        img = self.convert2arrayOriginal()
        h, w, _ = img.shape
        res = np.zeros(img.shape, np.uint8)  # creating blank mask for result
        l = 10  # the lower range of Hue we want
        u = 80 # the upper range of Hue we want
        mask = self.hsv(img, l, u)
        inv_mask = cv2.bitwise_not(mask)  # inverting mask
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        res1 = cv2.bitwise_and(img, img, mask=mask)  # region which has to be in color
        res2 = cv2.bitwise_and(gray, gray, mask=inv_mask)  # region which has to be in grayscale
        for i in range(3):
            res[:, :, i] = res2  # storing grayscale mask to all three slices
        img = cv2.bitwise_or(res1, res)  # joining grayscale and color region
        qimage = self.convertCvToQImage(img)
        q = QPixmap(qimage)
        self.imageLabel.set_imagePixmap(q)

    def __effect15(self):
        #MAGIC PENCIL
        img = self.convert2arrayOriginal()
        Kernel = np.array(([[10, 0, 0], [0, 0, 0], [0, 0, -10]]), np.float32) / 9
        DilateKernel = np.ones((2, 2), np.uint8)
        EdgeKernel = np.ones((3, 3), np.uint8)
        filterImg = cv2.filter2D(src=img, kernel=Kernel, ddepth=-1)
        edgeImg = cv2.Canny(img, 10, 300)
        filterImg = cv2.dilate(filterImg, DilateKernel)
        edgeImg = cv2.dilate(edgeImg, EdgeKernel)
        image = cv2.bitwise_and(filterImg, filterImg, mask=edgeImg)
        qimage = self.convertCvToQImage(image)
        q = QPixmap(qimage)
        self.imageLabel.set_imagePixmap(q)

    def __effect16(self):
        #SEPIA
        img = self.convert2arrayOriginal()
        kernel = np.float32([
            [0.272, 0.534, 0.131],
            [0.349, 0.686, 0.168],
            [0.393, 0.769, 0.189]])
        image = cv2.transform(img, kernel)
        qimage = self.convertCvToQImage(image)
        q = QPixmap(qimage)
        self.imageLabel.set_imagePixmap(q)

    def __effect17(self):
        #CHECKBOX BLACKMARK
        img = self.convert2arrayOriginal()
        h, w, _ = img.shape
        mask_1 = np.zeros([h, w, 1], dtype=np.uint8)
        mask_1[0:h, 0:w] = 255
        tile = 8
        for x in range(0, tile):
            for y in range(0, tile):
                if x % 2 == 0 and y % 2 == 0:
                    mask_1[int(h / tile) * y:int(h / tile) * (y + 1), int(w / tile) * x:int(w / tile) * (x + 1)] = 0
                elif x % 2 == 1 and y % 2 == 1:
                    mask_1[int(h / tile) * y:int(h / tile) * (y + 1), int(w / tile) * x:int(w / tile) * (x + 1)] = 0

        grayImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        edges = cv2.adaptiveThreshold(grayImg, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 5)
        mask_2 = cv2.bitwise_not(mask_1)
        checkbox = cv2.multiply(edges, mask_2)
        img1 = cv2.bitwise_and(img, img, mask=checkbox)
        img2 = cv2.bitwise_and(img, img, mask=mask_1)
        image = cv2.add(img1, img2)
        image = cv2.medianBlur(image, 5)
        qimage = self.convertCvToQImage(image)
        q = QPixmap(qimage)
        self.imageLabel.set_imagePixmap(q)

    def __effect18(self):
        #NOISE
        img = self.convert2arrayOriginal()
        h, w, _ = img.shape
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        thresh = 0.8
        for i in range(h):
            for j in range(w):
                if np.random.rand() <= thresh:
                    if np.random.randint(2) == 0:
                        img_gray[i, j] = min(img_gray[i, j] + np.random.randint(0, 64), 255)
                    else:
                        img_gray[i, j] = max(img_gray[i, j] - np.random.randint(0, 64), 0)
        qimage = self.convertCvToQImage(img_gray)
        q = QPixmap(qimage)
        self.imageLabel.set_imagePixmap(q)

    def __effect19(self):
        #NEGATIVE
        img = self.convert2arrayOriginal()
        blur = cv2.medianBlur(img, 5)
        image = cv2.bitwise_not(blur)
        qimage = self.convertCvToQImage(image)
        q = QPixmap(qimage)
        self.imageLabel.set_imagePixmap(q)

    def __effect20(self):
        #CIRCLE MOSAIC
        img = self.convert2arrayOriginal()
        h, w, _ = img.shape
        x = np.ones((10, 10), np.float32) / 100
        filter = cv2.filter2D(src=img, kernel=x, ddepth=-1)
        for i in range(0, h, 10):
            for j in range(0, w, 10):
                color = filter[i, j]
                color = (int(color[0]), int(color[1]), int(color[2]))
                cv2.ellipse(img, (j, i), (5, 5), 0, 0, 360, tuple(color), -1, cv2.LINE_AA)
        qimage = self.convertCvToQImage(img)
        q = QPixmap(qimage)
        self.imageLabel.set_imagePixmap(q)



def main():
    app = QApplication(sys.argv)
    win = Window()
    win.show()

    pm = QPixmap('Cat.jpg')
    win.imageLabel.set_imagePixmap(pm)
    return app.exec_()

if __name__ == '__main__':
    sys.exit(main())