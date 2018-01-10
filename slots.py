import numpy as np
import cv2
from PyQt5 import QtCore, QtGui, QtWidgets
import sys
import os
from mainwindow import Ui_MainWindow
from algorithm.LBF import LBF
from algorithm.region_grow import RegionGrow


class FarmWindow(QtWidgets.QMainWindow, Ui_MainWindow):
    def __init__(self, parent=None):
        super(FarmWindow, self).__init__(parent)
        self.setupUi(self)
        self.menubar.setNativeMenuBar(False)
        # window event connect
        self.stackedWidget.setCurrentIndex(0)
        self.level_set.clicked.connect(self.exec_level_set)
        self.comboBox.currentIndexChanged['int'].connect(self.change_method)
        self.region_grow.clicked.connect(self.exec_region_grow)
        self.delete_polygon.clicked.connect(self.exec_delete_polygon)
        self.gen_polygon.clicked.connect(self.exec_gen_polygon)
        self.actionopen.triggered.connect(self.open_file)
        self.actionreset.triggered.connect(self.reset_image)
        # define properties
        self.img_path = ''
        self.img_intersect = None
        self.img_intersect_bak = None
        self.img = None
        self.img_mask = None
        self.pixel_map = None
        self.img_clicked_buffer = []
        self.polygon = []
        self.level_set_label = None

    # define slot function
    def open_file(self):
        self.img_path, img_type = QtWidgets.QFileDialog.getOpenFileName(None, "Pick an image", '/', "*.png;;*.jpg;;*.jpeg;;*.bmp;;All Files (*)")

        print(self.img_path)
        if self.img_path:
            self.img_intersect = cv2.imread(self.img_path, cv2.IMREAD_ANYCOLOR)
            if len(self.img_intersect.shape) == 3:
                self.img_intersect = cv2.cvtColor(self.img_intersect, cv2.COLOR_BGR2RGB)
                self.img = cv2.cvtColor(self.img_intersect, cv2.COLOR_BGR2GRAY)
            else:
                self.img = self.img_intersect.copy()
            self.img_intersect_bak = self.img_intersect.copy()
            self.img_mask = np.zeros(self.img.shape, np.uint8)
            self.level_set_label = np.zeros(self.img.shape, np.uint8)

            self.pixel_map = QtGui.QPixmap(self.img_path)
            self.label_image.setPixmap(self.pixel_map)  # QPixmap=pixelmap 不行 见鬼了

    def reset_image(self):
        self.label_image.setPixmap(self.pixel_map)  # QPixmap=pixelmap 不行 见鬼了
        self.img_clicked_buffer = []
        self.img_intersect = self.img_intersect_bak.copy()

    def change_method(self):
        haha = self.comboBox.currentIndex()
        # print(haha)
        self.stackedWidget.setCurrentIndex(haha)
        self.img_clicked_buffer = []

    def exec_level_set(self):
        print('execute level set')
        time_step = float(self.time_step.text())
        epsilon = float(self.epsilon.text())
        mu = float(self.mu.text())
        lambda1 = float(self.lambda1.text())
        lambda2 = float(self.lambda2.text())
        sigma = float(self.sigma.text())
        c0 = float(self.c0.text())
        iter_num = int(self.iter_num.text())
        nu = float(self.nu.text())
        # 1.evolution
        if self.img is None:
            QtWidgets.QMessageBox.information(self,  # 使用infomation信息框
                                              "warnning",
                                              "please select a image first !",
                                              QtWidgets.QMessageBox.Yes)  # | QtWidgets.QMessageBox.No)
            return

        lbf = LBF(img=self.img, iter_num=iter_num, c0=c0, sigma=sigma,
                  nu=nu, mu=mu, lambda1=lambda1, lambda2=lambda2, epsilon=epsilon, time_step=time_step)
        bw = lbf.level_set_evolution()

        _, contours1, _ = cv2.findContours(bw.copy(), mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_NONE)
        _, contours2, _ = cv2.findContours(255 - bw, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_NONE)

        contours1 = sorted(contours1, key=lambda x: x.shape[0])
        for ii in range(len(contours1)):
            if contours1[ii].shape[0] > 20:
                del contours1[0:ii]
                break
        contours2 = sorted(contours2, key=lambda x: x.shape[0])
        for ii in range(len(contours2)):
            if contours2[ii].shape[0] > 20:
                del contours2[0:ii]
                break

        cont1 = np.zeros(self.img.shape[0:2], dtype=np.uint8)
        cont2 = np.zeros(self.img.shape[0:2], dtype=np.uint8)

        for iter1 in range(len(contours1)):
            # print(iter1)
            tp = contours1[iter1]
            tp = np.squeeze(tp, axis=1)
            contours1[iter1] = tp
            cont1[tp[:, 1], tp[:, 0]] = 255
        for iter1 in range(len(contours2)):
            tp = contours2[iter1]
            tp = np.squeeze(tp, axis=1)
            contours2[iter1] = tp
            cont2[tp[:, 1], tp[:, 0]] = 255
        cv2.imwrite('/Users/whyguu/Desktop/huhu.jpg', cont1)
        cv2.imwrite('/Users/whyguu/Desktop/xixi.jpg', cont2)
        cv2.imwrite('/Users/whyguu/Desktop/haha.jpg', bw)

        for it in range(len(contours1)):
            cv2.fillPoly(self.level_set_label, [contours1[it]], color=it+1)
        # np.savez('xixi.npz', cn1=contours1[3], cn2=contours1[5])
        print('level_Set done !')
        cv2.imwrite('mimi.jpg', self.level_set_label)

    def exec_region_grow(self):
        # region grow and 划出结果
        if len(self.img_clicked_buffer) == 0:
            QtWidgets.QMessageBox.information(self, "warnning", "please select points first !", QtWidgets.QMessageBox.Yes)
            return
        region = self.img_clicked_buffer.copy()
        self.img_clicked_buffer.clear()
        std_decay = float(self.std_decay.text())
        print('execute region grow...')
        con = RegionGrow.region_grow(region, self.img, self.img_mask, std_decay)
        print('region grow done !')
        # #################
        for it in range(len(con)-1):
            cv2.line(self.img_intersect, (con[it][0], con[it][1]), (con[it+1][0], con[it+1][1]),
                     (123, 34, 189))
        cv2.line(self.img_intersect, (con[0][0], con[0][1]), (con[-1][0], con[-1][1]),
                 (123, 34, 189))
        # #################
        pixmap = self.fromNumpy2Pixmap(self.img_intersect)
        self.label_image.clear()
        self.label_image.setPixmap(pixmap)

        self.img_mask = np.zeros(self.img.shape, np.uint8)
        self.polygon.append(con)
        print(con)

    def exec_gen_polygon(self):
        # from numpy to QPixmap
        qimg = QtGui.QImage(self.img.data, self.img.shape[1], self.img.shape[0], self.img.shape[1],
                            QtGui.QImage.Format_Indexed8)
        pixmap = QtGui.QPixmap.fromImage(qimg)
        self.label_image.setPixmap(pixmap)

    def exec_delete_polygon(self):
        pass

    def mouseReleaseEvent(self, QMouseEvent):
        if QMouseEvent.button() != QtCore.Qt.LeftButton:
            return
        qpoint = QMouseEvent.globalPos()
        # print(qpoint)

        # label_image process
        lb_img_pnt = self.label_image.mapFromGlobal(qpoint)
        lb_img_width = self.label_image.contentsRect().width()
        lb_img_height = self.label_image.contentsRect().height()
        # assert in the range of label
        if lb_img_pnt.x() in range(0, lb_img_width) and lb_img_pnt.y() in range(0, lb_img_height):
            if self.img is None:
                QtWidgets.QMessageBox.information(self,  # 使用infomation信息框
                                                  "warnning",
                                                  "please select a image first !",
                                                  QtWidgets.QMessageBox.Yes)  # | QtWidgets.QMessageBox.No)
                return
            img_offset_x = (lb_img_width - self.label_image.pixmap().rect().width()) // 2
            img_offset_y = (lb_img_height - self.label_image.pixmap().rect().height()) // 2
            img_x = lb_img_pnt.x() - img_offset_x
            img_y = lb_img_pnt.y() - img_offset_y
            # assert in the range of image
            if img_x not in range(0, self.label_image.pixmap().rect().width()) or img_y not in range(0, self.label_image.pixmap().rect().height()):
                return

            self.img_clicked_buffer.append((img_y, img_x))
            r_min = max(img_y - 1, 0)
            r_max = min(img_y + 1, self.img.shape[0] - 1)
            c_min = max(img_x - 1, 0)
            c_max = min(img_x + 1, self.img.shape[1] - 1)
            for tp_r in range(r_min, r_max + 1):
                for tp_c in range(c_min, c_max + 1):
                    self.img_intersect[tp_r, tp_c] = 255
            # repaint

            pixmap = self.fromNumpy2Pixmap(self.img_intersect)
            self.label_image.clear()
            self.label_image.setPixmap(pixmap)

            print(img_x)
            print(img_y)

    def fromNumpy2Pixmap(self, img):
        # from numpy to QPixmap
        if len(img.shape) == 3:
            h, w, c = img.shape
            img_format = QtGui.QImage.Format_RGB888
        else:
            h, w = img.shape
            c = 1
            img_format = QtGui.QImage.Format_Indexed8
        qimg = QtGui.QImage(img.data, w, h, c * w, img_format)
        pixmap = QtGui.QPixmap.fromImage(qimg)
        return pixmap


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    farm = FarmWindow()
    farm.show()
    sys.exit(app.exec_())
