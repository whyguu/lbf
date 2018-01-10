import numpy as np
import cv2
from scipy.ndimage import filters
# from osgeo import gdal, ogr, osr
# from shapely.geometry import Polygon
import os
import shutil
import argparse


class LBF(object):

    def __init__(self, img, iter_num=500, c0=2, sigma=3.0, nu=0.003, mu=1.0, lambda1=1.0, lambda2=1.0, epsilon=2.0, time_step=0.1):
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.iter_num = iter_num
        self.nu = nu*255*255
        self.mu = mu
        self.epsilon = epsilon
        self.sigma = sigma
        self.time_step = time_step
        self.c0 = c0
        self.img = img.copy()

        self.phi = None
        self.img_kernel = None
        self.one_kernel = None
        self.kernel_size = round(2*self.sigma)*2+1

    def public_data(self):
        r, c = self.img.shape
        self.phi = np.ones(self.img.shape)  # level set function
        # _, self.phi = cv2.threshold(self.img, 0, 1, cv2.THRESH_OTSU)
        cv2.circle(self.phi, (c//2, r//2), radius=4, color=0, thickness=-1)

        self.phi = self.c0 * 2*(self.phi-0.5)
        self.img = np.float64(self.img)
        self.phi = np.float64(self.phi)

        cv2.imwrite('init.bmp', self.phi*255)
        '''
        with open("test.csv", "w") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerows(self.phi)
        '''
        self.img_kernel = cv2.GaussianBlur(self.img, (self.kernel_size, self.kernel_size), self.sigma, borderType=cv2.BORDER_CONSTANT)
        self.one_kernel = cv2.GaussianBlur(np.ones(self.img.shape, np.float64), (self.kernel_size, self.kernel_size), self.sigma, borderType=cv2.BORDER_CONSTANT)

    def level_set_evolution(self):
        self.public_data()

        for i in range(self.iter_num):
            #  Neumann Boundary Condition
            self.neum_bound_cond()

            #  dirac
            dirac_phi = self.dirac()

            #  local binary fit
            f1, f2 = self.local_bin_fit()

            #  data force
            lbf = self.data_force(f1, f2)

            #  curvature central
            curv = self.curvature_central()

            # each item
            area_term = -dirac_phi * lbf
            len_term = self.nu * dirac_phi * curv

            laplace_operator_phi = filters.laplace(self.phi)  # self.phi laplace operator
            penalty = self.mu * (laplace_operator_phi - curv)

            self.phi = self.phi + self.time_step * (area_term + penalty + len_term)

        # binary phi and then return
        rtn_phi = np.uint8(self.phi > 0)

        return rtn_phi*255

    def neum_bound_cond(self):
        r, c = self.phi.shape
        self.phi[np.ix_([0, -1], [0, -1])] = self.phi[np.ix_([2, -3], [2, -3])]
        self.phi[np.ix_([0, -1], np.arange(1, c-1))] = self.phi[np.ix_([2, -3], np.arange(1, c-1))]
        self.phi[np.ix_(np.arange(1, r-1), [0, -1])] = self.phi[np.ix_(np.arange(1, r-1), [2, -3])]

    def heaviside(self):
        heav_phi = 0.5 * (1+(2/3.1415926)*np.arctan(self.phi/self.epsilon))
        return heav_phi

    def dirac(self):
        dirac_phi = (self.epsilon/3.1415926)/(self.epsilon*self.epsilon+self.phi*self.phi)
        return dirac_phi

    def local_bin_fit(self):
        heav_phi = self.heaviside()
        image = self.img * heav_phi
        c1 = cv2.GaussianBlur(heav_phi, (self.kernel_size, self.kernel_size), self.sigma, borderType=cv2.BORDER_CONSTANT)
        c2 = cv2.GaussianBlur(image, (self.kernel_size, self.kernel_size), self.sigma, borderType=cv2.BORDER_CONSTANT)
        f1 = c2 / c1
        f2 = (self.img_kernel-c2) / (self.one_kernel-c1)
        return f1, f2

    def data_force(self, f1, f2):
        s1 = self.lambda1 * f1 * f1 - self.lambda2 * f2 * f2
        s2 = self.lambda1 * f1 - self.lambda2 * f2
        gs1 = cv2.GaussianBlur(s1, (self.kernel_size, self.kernel_size), self.sigma, borderType=cv2.BORDER_CONSTANT)
        gs2 = cv2.GaussianBlur(s2, (self.kernel_size, self.kernel_size), self.sigma, borderType=cv2.BORDER_CONSTANT)
        lbf = (self.lambda1-self.lambda2) * self.one_kernel * self.img * self.img + gs1 - 2.0 * self.img * gs2

        return lbf

    def curvature_central(self):
        ux, uy = np.gradient(self.phi)
        norm = np.sqrt(ux*ux + uy*uy + 1e-10)
        nx = ux / norm
        ny = uy / norm

        nxx, _ = np.gradient(nx)
        _, nyy = np.gradient(ny)

        curv = nxx + nyy
        return curv


class MyAlgorithm(object):
    def __init__(self):
        pass

    @staticmethod
    def douglas(con):
        def douglas_impl(dog, con, mark, start, end):
            # 函数出口
            if start == end - 1:
                return
            # 计算直线 Ax+By+C=0
            x1, y1 = con[start, 0]
            x2, y2 = con[end, 0]
            A = y2 - y1
            B = x1 - x2
            C = x2 * y1 - x1 * y2
            # 计算个点到直线的距离
            max_dist = 0
            for i in range(start + 1, end - 1):
                tp_dist = abs(A * con[i][0][0] + B * con[i][0][1] + C) / np.sqrt(pow(A, 2) + pow(B, 2))
                if tp_dist > max_dist:
                    index = i
                    max_dist = tp_dist

            if max_dist > dog:
                douglas_impl(dog, con, mark, start, index)
                douglas_impl(dog, con, mark, index, end)
            else:
                mark[start + 1:end, 0] = 0

        # con: a list which are generated by function cv2.findContours() of ndarrays
        dog = 1  # douglas parameter
        for i in range(len(con)):
            start = 0
            end = len(con[i]) - 1
            mark = np.ones((len(con[i]), 1))
            douglas_impl(dog, con[i], mark, start, end)
            con[i] = con[i][mark[:, 0] == 1]
        # return con

    @staticmethod
    def gen_shape_file(file_path, img_path, polygon, layer_name):
        # polygon: a list which are generated by function cv2.findContours() of ndarrays
        if not file_path or not img_path:
            print('the shape file path or the image path may be not accessible !')
            return
        # if os.path.exists(file_path):
        if os.path.exists(file_path) and len(os.listdir(file_path)) > 1:
            shutil.rmtree(file_path)

        driver = ogr.GetDriverByName('Esri Shapefile')
        ds = driver.CreateDataSource(file_path)

        layer = ds.CreateLayer(layer_name, None, ogr.wkbPolygon)
        # Add one attribute
        layer.CreateField(ogr.FieldDefn('id', ogr.OFTInteger))
        defn = layer.GetLayerDefn()
        dataset = gdal.Open(img_path)
        trans = dataset.GetGeoTransform()
        no_geo_info = 0
        img_row = 0
        if trans == (0.0, 1.0, 0.0, 0.0, 0.0, 1.0):
            img_row = dataset.RasterYSize
            no_geo_info = 1
            print('the chosen image may not contain geometry info , shape file will be useless !')  ###########################
        feat = ogr.Feature(defn)
        for iter1 in range(len(polygon)):
            tp_poly = polygon[iter1][:, 0, :]
            poly = []
            if tp_poly.shape[0] < 3:
                continue
            for iter2 in range(tp_poly.shape[0]):
                # 此处应该得到 tp_poly[iter2, :] 对应的 地理坐标  ########################################
                cvx = tp_poly[iter2, 0]  # opencv 中的 x
                cvy = tp_poly[iter2, 1]  # opencv 中的 y
                if no_geo_info:
                    cvy = img_row - tp_poly[iter2, 1]  # opencv 中的 y

                px = trans[0] + cvx * trans[1] + cvy * trans[2]
                py = trans[3] + cvx * trans[4] + cvy * trans[5]

                poly.append((px, py))
            poly.append(poly[0])
            poly = Polygon(poly)

            # Create a new feature (attribute and geometry)
            feat.SetField('id', iter1)

            # Make a geometry, from Shapely object
            geom = ogr.CreateGeometryFromWkb(poly.wkb)
            feat.SetGeometry(geom)

            layer.CreateFeature(feat)
        feat = geom = None  # destroy these
        # Save and close everything
        ds = layer = feat = geom = None


def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-image_path', type=str, required=True)
    parser.add_argument('-file_path', type=str, required=True)
    return parser


if __name__ == "__main__":
    # path = './mypictures/test01.jpg'
    # path = '/Users/whyguu/Desktop/WechatIMG10.jpeg'
    parser = arg_parser()
    args = parser.parse_args()
    path = args.image_path
    file_path = args.file_path

    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    cont1 = []
    cont2 = []
    # roi_win = cv2.namedWindow('select ROI', cv2.WINDOW_NORMAL)

    # 1.evolution
    lbf = LBF(img=img, iter_num=500)
    bw = lbf.level_set_evolution()

    # cv2.imwrite('rst.bmp', bw)
    # 2.find contours
    _, contours1, _ = cv2.findContours(bw.copy(), mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_NONE)
    _, contours2, _ = cv2.findContours(255 - bw, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_NONE)

    cont1 += contours1
    cont2 += contours2
    # 4.generate shape file
    MyAlgorithm.gen_shape_file(os.path.join(file_path, 'whitefile'), path, cont1, layer_name='white')
    MyAlgorithm.gen_shape_file(os.path.join(file_path, 'blackfile'), path, cont2, layer_name='black')

    print('finished!')


