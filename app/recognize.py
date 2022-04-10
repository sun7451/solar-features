#!usr/bin/env python
# -*- coding:utf-8 -*-
"""   
Author          Sun
Create          2022-04-10 6:29 PM
"""
import os
from cProfile import label
import numpy as np
from cv2 import cv2
from skimage import feature, morphology
from skimage.measure import label, regionprops
from skimage.filters import threshold_otsu
import matplotlib.patches as mpatches
import numpy.ma as ma
from app.prep import Prep
from common.log import Log

logger = Log.build_logger('{}{}'.format(Log.get_base_dir(), '/log_dir/log.log'))


class Recognize(object):

    @staticmethod
    def have_cloud(name):
        img0 = Prep.readim2gray(name)

        img_smooth = cv2.medianBlur(img0, 13)  # give a smooth

        clean_image = Prep.cloud_detection(img_smooth)  # this give a cloud detection
        # clean_image 为1，没有云，返回False
        if clean_image:
            return False
        else:
            return True

    @staticmethod
    def detect_filament(name: str, ifcheck: int):
        img0 = Prep.readim2gray(name)

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
    def filament_data(name, corrected, labelim):
        interesting = []
        filament_coords = []
        filament_data = list()
        figure_data = [corrected]
        rect_list = list()

        for number, region in enumerate(regionprops(labelim, intensity_image=corrected)):
            filament_id = "%s_%04d" % (name, number)
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

                rect_list.append([filament_id, rect])

                roi = corrected[minr - 5:minr + maxr - minr + 5, minc - 5:minc + maxc - minc + 5]
                # this gives the coords for each filament
                filament_coords.append(region.coords)
                # this gives the rectangle to tag each filament
                interesting.append(roi)
                # go to JSON

                data = {
                    "id": filament_id,
                    "centroid": region.centroid,
                    "area": float(region.area_filled),
                    "length": region.axis_major_length,
                    "width": region.axis_minor_length,
                    "outlines": [[int(x), int(y)] for x, y in region.coords]
                }
                filament_data.append(data)

        if filament_data:
            figure_data.append(rect_list)

            return figure_data, filament_data

    def get_filament_datas(self, input_dir, image_nums=3):
        filament_datas = list()
        figure_datas = list()
        i = 0
        file_names = os.listdir(input_dir + "/images")
        for file_name in sorted(file_names):
            if file_name.startswith("20") and file_name.endswith(".png"):
                image_dir = input_dir + "/images/" + file_name
                try:
                    name = file_name[0:14]

                    if self.have_cloud(image_dir):
                        logger.info("{} have cloud, skip it".format(image_dir))
                        continue

                    labelim, _, corrected = self.detect_filament(image_dir, ifcheck=1)

                    figure_data, filament_data = self.filament_data(name, corrected, labelim)

                    filament_datas.append([name, filament_data])
                    figure_datas.append([name, figure_data])
                    # -------------------------
                    i = i + 1
                    if i >= image_nums:
                        break

                except Exception as e:
                    logger.error("name {} is pass {}".format(file_name, e))

        return filament_datas, figure_datas
