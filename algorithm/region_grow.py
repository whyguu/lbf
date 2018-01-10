# -*- coding: utf-8 -*-
"""
Created on Mon Aug  7 16:38:11 2017

@author: why
"""

import numpy as np
import cv2


class RegionGrow(object):
    def __init__(self):
        pass

    @staticmethod
    def gen_seeds(region, img_mask):

        if len(region) >= 3:
            # 所选点大于等于3 执行求重心 然后选取种子点
            RegionGrow.generate_seed_points(region, img_mask)
        else:
            # 所选点小于3 则在周围3*3区域内采样种子点
            for i in range(len(region)):
                r_point = region[i]
                r_min = max(r_point[0] - 1, 0)
                r_max = min(r_point[0] + 1, img_mask.shape[0] - 1)
                c_min = max(r_point[1] - 1, 0)
                c_max = min(r_point[1] + 1, img_mask.shape[1] - 1)
                for r, c in zip([r_min, r_max, r_point[0], r_point[0]], [r_point[1], r_point[1], c_min, c_max]):
                    if img_mask[r, c] != 255:
                        region.append((r, c))
                        img_mask[r, c] = 255

    @staticmethod
    def gravity_center1(points):
        length = len(points)
        area_points = np.zeros((length - 2, 3))
        x1, y1 = points[0]
        for i in range(length - 2):
            x2, y2 = points[i + 1]
            x3, y3 = points[i + 2]

            tp_r = (x1 + x2 + x3) / 3.0
            tp_c = (y1 + y2 + y3) / 3.0
            tp_area = 0.5 * np.abs((x1 * y2 - x2 * y1) + (x2 * y3 - x3 * y2) + (x3 * y1 - x1 * y3))

            area_points[i, :] = [tp_r, tp_c, tp_area]
        # print area_points, 'in gravity_center1'
        mass = np.sum(area_points[:, 2])
        r_mass = np.dot(area_points[:, 0], area_points[:, 2])
        c_mass = np.dot(area_points[:, 1], area_points[:, 2])
        r = r_mass / mass
        c = c_mass / mass
        return r, c

    @staticmethod
    def get_gauss_para(region, img):
        # transfer total  total2 can speed up calculation
        sumx = 0.0
        sumx2 = 0.0
        length = len(region)
        if length == 0:
            return
        for i in range(length):
            sumx += img[region[i]]
            sumx2 += pow(img[region[i]], 2)
        mean = sumx / length
        std = np.sqrt((sumx2 - pow(mean, 2) * length) / (length - 1))
        return mean, std, sumx2, sumx

    @staticmethod
    def update_gauss_para(sumx2, sumx, region, counter, img):
        sumx2 += pow(img[region], 2)
        sumx += img[region]

        mean = 1.0 * sumx / counter
        try:
            std = np.sqrt((sumx2 - pow(mean, 2) * counter) / (counter - 1))
        except Exception:
            print('(sumx2 - pow(mean, 2) * counter', (sumx2 - pow(mean, 2) * counter))
            print('counter - 1)', counter - 1)
        return mean, std, sumx2, sumx

    @staticmethod
    def region_grow(region, img, img_mask, std_decay):
        count = len(region)
        print('count=', count)
        RegionGrow.gen_seeds(region, img_mask)
        counter = len(region)
        print('generate seeds successfully !')
        print('get gaussion para...')
        mean, std, sumx2, sumx = RegionGrow.get_gauss_para(region, img)
        print('get gaussion para done !')
        print(mean, std, sumx, sumx2)
        std_ratio = 2.0
        reg_min = mean - std_ratio * std
        reg_max = mean + std_ratio * std
        print('reg_max, reg_min:', reg_max, reg_min)
        # region grow
        i = 0
        print('enter while .... ')
        while len(region):
            rpoint = region.pop(0)
            # print(rpoint)
            i += 1
            r_min = max(rpoint[0] - 1, 0)
            r_max = min(rpoint[0] + 1, img.shape[0] - 1)
            c_min = max(rpoint[1] - 1, 0)
            c_max = min(rpoint[1] + 1, img.shape[1] - 1)

            for r, c in zip([r_min, r_max, rpoint[0], rpoint[0]], [rpoint[1], rpoint[1], c_min, c_max]):
                print('img[r, c] = ', img[r, c])
                if img[r, c] > reg_min and img[r, c] < reg_max and img_mask[r, c] != 255:
                    region.append((r, c))
                    img_mask[r, c] = 255
                    counter += 1
                    print('sdjhkskd',counter, img_mask[r, c])
                    if i < count * img.shape[0] * img.shape[1] / std_decay:

                        mean, std, sumx2, sumx = RegionGrow.update_gauss_para(sumx2, sumx, region[-1], counter, img)
                        ww = std_ratio * std
                        reg_min = mean - ww
                        reg_max = mean + ww
                    elif std_ratio > 2 / np.log(std):
                        std_ratio -= 450.0 / img.shape[0] / img.shape[1]
                        # std_ratio = 1.5
                        ww = std_ratio * std
                        reg_min = mean - ww
                        reg_max = mean + ww
        print('out while !')
        cv2.imwrite('hehe.jpg', img_mask)
        con = RegionGrow.draw_region(img_mask)
        return con

    @staticmethod
    def generate_seed_points(points, img_mask):
        center = RegionGrow.gravity_center1(points)
        r, c = center
        img_len = np.sqrt(img_mask.shape[0] * img_mask.shape[1]) / 40
        for i in range(len(points)):
            point = points[i]
            if center[0] == point[0]:
                tp = np.abs(center[1] - point[1])
                if tp > img_len * 2:
                    l = int(tp / img_len)
                tpmin = min(center[1], point[1])
                for ii in range(1, l):

                    if img_mask[(int(point[0]), int(tpmin + tp * ii / l))] != 255:
                        points.append((int(point[0]), int(tpmin + tp * ii / l)))
                        img_mask[points[-1]] = 255
                continue

            k, b = RegionGrow.line_parameter(center, point)
            rx = min(r, point[0])
            rl = np.abs(r - point[0])
            p_dist = np.linalg.norm([rl, np.abs(c - point[1])])
            if p_dist > img_len * 2:
                l = int(p_dist / img_len)
            else:
                l = 3
            for ii in range(1, l):
                nr = rx + rl / l * ii
                nc = k * nr + b
                if img_mask[(int(nr), int(nc))] != 255:
                    points.append((int(nr), int(nc)))
                    img_mask[points[-1]] = 255
        if img_mask[(int(r), int(c))] != 255:
            points.append((int(r), int(c)))
            img_mask[points[-1]] = 255

    @staticmethod
    def draw_region(img_mask):
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        img_mask = cv2.morphologyEx(img_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        cv2.imwrite('haha.jpg', img_mask)
        # contour
        con = RegionGrow.get_contour(img_mask)
        print('con.shape = ', con.shape)
        # douglas(con)
        con = RegionGrow.douglas(con)
        print('con.shape = ', con.shape)
        return con

    @staticmethod
    def get_contour(img_mask):
        # 求最外层的边界
        _, con, _ = cv2.findContours(img_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_L1)  # cv2.CHAIN_APPROX_TC89_KCOS
        # 可能求出多个边界 只要最大的那个
        max_val = con[0].shape[0]
        max_ind = 0
        for i in range(1, len(con)):
            if max_val < con[i].shape[0]:
                max_ind = i
                max_val = con[i].shape[0]
        con = con[max_ind][:, 0, :]
        return con

    @staticmethod
    def douglas_impl(dog, con, mark, start, end):
        # 函数出口
        if start == end - 1:
            return
        # 计算直线 Ax+By+C=0
        x1, y1 = con[start]
        x2, y2 = con[end]
        A = y2 - y1
        B = x1 - x2
        C = x2 * y1 - x1 * y2
        # 计算个点到直线的距离
        max_dist = 0
        for i in range(start + 1, end - 1):
            tp_dist = abs(A * con[i][0] + B * con[i][1] + C) / np.sqrt(pow(A, 2) + pow(B, 2))
            if tp_dist > max_dist:
                index = i
                max_dist = tp_dist

        if max_dist > dog:
            RegionGrow.douglas_impl(dog, con, mark, start, index)
            RegionGrow.douglas_impl(dog, con, mark, index, end)
        else:
            mark[start + 1:end, 0] = 0

    @staticmethod
    def douglas(con):
        dog = 15  # douglas parameter
        start = 0
        end = len(con) - 1
        mark = np.ones((len(con), 1))
        RegionGrow.douglas_impl(dog, con, mark, start, end)
        con = con[mark[:, 0] == 1, :]
        return con

    @staticmethod
    def line_parameter(point1, point2):
        x1, y1 = point1
        x2, y2 = point2
        k = 1.0 * (y2 - y1) / (x2 - x1)
        b = y1 - k * x1
        return k, b
