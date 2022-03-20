#!usr/bin/env python
# -*- coding:utf-8 -*-
"""   
Author          Sun
Create          2022-03-14 9:37 PM
"""
import math
import os

from cv2 import cv2
import numpy as np
from skimage import morphology
import matplotlib.pyplot as plt
from prep import Prep
from skimage import feature
from skimage.filters import threshold_otsu
from skimage.measure import label, regionprops
import numpy.ma as ma
import matplotlib.patches as mpatches
import logging
import json
from skimage.metrics import structural_similarity
from collections import defaultdict
from typing import Dict, List

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class Detecting(object):

    def __init__(self):
        self.__doc__ = ""

    @staticmethod
    def detect_filament(name: str, ifcheck: int):
        img0 = Prep.readim2gray(name)

        img_smooth = cv2.medianBlur(img0, 13)  # give a smooth
        cloud = Prep.cloud_detection(img_smooth)  # this give a cloud detection

        corrected, center, Rsun = Prep().remove_limbdark(img0)

        corrected = Prep.intensity_gray(corrected)
        # -------------------------------------
        # give the threshold for the filament dentection
        blur = 2 * corrected - cv2.GaussianBlur(corrected, (21, 21), 0)  # remove the opacity of clouds
        threshold = -0.35 * np.std(blur) + np.median(blur)  # define the global threshold
        ret3, threshim = cv2.threshold(blur, threshold, 255, cv2.THRESH_BINARY_INV)  # give a filterd image
        threshim = Prep.zero_limb(threshim, center, Rsun)  # remove the limb
        # ---------------------remove all sporadic features------------------------
        remain_regions = []
        # Features as selected by the surface criteria
        # This can discard sunspots and other small round features
        im_label = label(threshim, connectivity=2)
        for pts in regionprops(im_label):
            if pts.area >= 20 and pts.axis_major_length >= 10:
                e = pts.axis_major_length / pts.axis_minor_length
                if e > 1.2:
                    remain_regions.append(pts)
        gray = threshim.copy()
        mask = np.zeros(np.shape(gray))
        for k in remain_regions:
            # print('area,length,width=',k.area,k.axis_major_length,k.axis_minor_length)
            x_c = np.zeros(len(k.coords))
            y_c = np.zeros(len(k.coords))

            for i in np.arange(len(k.coords)):
                x_c[i] = k.coords[i][0]
                y_c[i] = k.coords[i][1]
                mask[x_c.astype(int), y_c.astype(int)] = True
        binary = mask.copy()
        # canny edge detection
        edges2 = feature.canny(binary, sigma=1)
        # make a fat kernel
        kernel = np.ones((5, 5), np.uint8)
        # Dilation with this kernel
        dilation = morphology.binary_dilation(edges2, kernel)
        # Erosion to recover the original shape
        erosion = morphology.binary_erosion(dilation, kernel)
        # Now we want to tag the structures that have a surface bigger than a threshold to only keep filaments and discard spots

        image = erosion.copy()
        # apply threshold
        thresh = threshold_otsu(image)
        # print('threshold_otsu=',thresh)
        bw = morphology.closing(image > thresh, morphology.square(3))
        # remove artifacts connected to image border
        # label image regions
        labelim = label(bw, connectivity=1)
        gray = corrected.copy()
        result = ma.masked_where(mask, gray)

        if ifcheck is False:
            ifcheck = 0
        if ifcheck == 1:
            imgs = [corrected, threshim, binary, result]
            labels = ['corrected', 'global threshold', 'binary', 'result']
            # self.multi_show(imgs, labels)
        # labelim presents the labeled image
        # result presents the corrected Ha image + tags
        # corrected presents the ccorrected Ha image
        return labelim, result, corrected

    @staticmethod
    def multi_show(imgs, labels):
        fig = plt.figure(figsize=(7, 7))
        for i in range(len(imgs)):
            img = imgs[i]
            title = str(i + 1)
            plt.subplot(2, 2, i + 1)
            plt.imshow(img, cmap='gray')
            plt.title(labels[i], fontsize=8)
            # plt.colorbar()
            # plt.xticks([])
            # plt.yticks([])
        plt.tight_layout()
        plt.show()
        return 0

    def get_filament_data(self, name, corrected, labelim):
        data = list()
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.imshow(corrected, cmap='gray')
        interesting = []
        filament_coords = []
        filament_list = list()
        for number, region in enumerate(regionprops(labelim, intensity_image=corrected)):
            if region.area >= 10:
                # this can give an example for information extract for each filament
                logger.info('centroid,area,length,width= {}{}{}{}'.format(region.centroid,
                                                                          region.area_filled,
                                                                          region.axis_major_length,
                                                                          region.axis_minor_length))
                # draw rectangle around segmented coins
                minr, minc, maxr, maxc = region.bbox
                rect = mpatches.Rectangle((minc - 5, minr - 5), maxc - minc + 10, maxr - minr + 10,
                                          fill=False, edgecolor='white', linewidth=1)
                ax.add_patch(rect)
                roi = corrected[minr - 5:minr + maxr - minr + 5, minc - 5:minc + maxc - minc + 5]
                # this gives the coords for each filament
                filament_coords.append(region.coords)
                # this gives the rectangle to tag each filament
                interesting.append(roi)
                # go to JSON

                data = {
                    "id": number,
                    "centroid": region.centroid,
                    "area": float(region.area_filled),
                    "length": region.axis_major_length,
                    "width": region.axis_minor_length,
                    "outlines": [[int(x), int(y)] for x, y in region.coords]
                }
                filament_list.append(data)

        # if filament_list:
        # self.write_to_json_file(name[0:14], filament_list)
        # ax.set_axis_off()
        # plt.tight_layout()
        # plt.show()
        return [name[0:14], filament_list]

    def get_data(self):
        # name = './images/20150302164934Ch.png'
        file_names = os.listdir("./images")
        filament_datas = list()
        for name in sorted(file_names):
            if name.startswith("20") and name.endswith(".png"):
                image_dir = "./images/" + name

                try:
                    labelim, result, corrected = self.detect_filament(image_dir, ifcheck=1)
                    filament_datas.append(self.get_filament_data(name, corrected, labelim))
                except Exception as e:
                    logger.error("name {} is pass {}".format(name, e))
        return filament_datas

    @staticmethod
    def write_to_json_file(name: str, filament_data: List[dict]):
        with open("./json_data/{}.json".format(name), "w") as f:
            f.write(json.dumps(filament_data) + "\n")
        return 0

    def track_filament(self, filament_datas):
        filament_datas = sorted(filament_datas, key=lambda x: x[0])
        n = len(filament_datas)
        data = list()
        for i in range(n - 1):
            # 两两比较
            name1, data1 = filament_datas[i]
            name2, data2 = filament_datas[i+1]
            match = self.track(data1, data2)
            data.append([(name1, name2), match])
        return data

    def track(self, data1, data2):
        match = list()
        for item1 in data1:
            x1, y1 = item1.get("centroid")
            feature1 = np.array([item1.get("area"), item1.get("length"), item1.get("width")])

            for item2 in data2:
                x2, y2 = item2.get("centroid")
                feature2 = np.array([item2.get("area"), item2.get("length"), item2.get("width")])
                t = self.cos_theta(feature1, feature2)

                # 追踪判断
                if (x1 <= x2) and abs(y2 - y1) <= 10 and t > 0.9:
                    match.append((item1.get("id"), item2.get("id")))
                    break
        return match

    @staticmethod
    def cos_theta(array1, array2):
        norm1 = math.sqrt(sum(list(map(lambda x: math.pow(x, 2), array1))))
        norm2 = math.sqrt(sum(list(map(lambda x: math.pow(x, 2), array2))))
        return sum([array1[i] * array2[i] for i in range(0, len(array1))]) / (norm1 * norm2)

    def start(self):
        filament_datas = self.get_data()
        self.track_filament(filament_datas)


if __name__ == '__main__':
    Detecting().start()
