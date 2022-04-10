#!usr/bin/env python
# -*- coding:utf-8 -*-
"""
Author          Sun
Create          2022-03-14 9:37 PM
"""
import math
import os
import numpy as np
import matplotlib.pyplot as plt
from app.recognize import Recognize
import json
from skimage.metrics import structural_similarity
from collections import defaultdict
from common.log import Log

logger = Log.build_logger('{}{}'.format(Log.get_base_dir(), '/log_dir/log.log'))


class Track(object):

    def __init__(self, input_dir="data/input", output_dir="data/output", similarity_threshold=0.6):
        super().__init__()
        self.__doc__ = ""
        self.similarity_threshold = similarity_threshold
        self.input_dir = input_dir
        self.output_dir = output_dir

    def show_figure_data(self, figure_data, **kwargs):
        os.makedirs(self.output_dir + "/image_data", exist_ok=True)

        name, (corrected, rects) = figure_data
        fig, ax = plt.subplots(figsize=(10, 6))

        ax.imshow(corrected, cmap='gray')
        for filament_id, rect in rects:
            ax.add_patch(rect)

            # 非法id排查
            if len(filament_id) > 10:
                continue

            plt.annotate(filament_id, xy=(rect.get_x(), rect.get_y()))

        plt.title(name)
        plt.tight_layout()

        if kwargs.get("is_save_image"):
            plt.savefig(self.output_dir + "/image_data/{}.png".format(name))

        if kwargs.get("is_show_image"):
            plt.show()
        return 0

    @staticmethod
    def cos_theta(array1, array2):
        # 计算向量的余弦值，也就是相似度
        norm1 = math.sqrt(sum(list(map(lambda x: math.pow(x, 2), array1))))
        norm2 = math.sqrt(sum(list(map(lambda x: math.pow(x, 2), array2))))
        return sum([array1[i] * array2[i] for i in range(0, len(array1))]) / (norm1 * norm2)

    @staticmethod
    def outlines_similarity(outlines1, outlines2):
        outlines_array1 = np.array(outlines1)
        outlines_array2 = np.array(outlines2)

        minx1, maxx1 = min(outlines_array1[:, 0]), max(outlines_array1[:, 0])
        miny1, maxy1 = min(outlines_array1[:, 1]), max(outlines_array1[:, 1])

        minx2, maxx2 = min(outlines_array2[:, 0]), max(outlines_array2[:, 0])
        miny2, maxy2 = min(outlines_array2[:, 1]), max(outlines_array2[:, 1])

        # 平移到原点
        outlines_array1[:, 0] = outlines_array1[:, 0] - minx1
        outlines_array1[:, 1] = outlines_array1[:, 1] - miny1

        outlines_array2[:, 0] = outlines_array2[:, 0] - minx2
        outlines_array2[:, 1] = outlines_array2[:, 1] - miny2

        # 获取等大的图形
        m = max([maxx1 - minx1 + 1, maxx2 - minx2 + 1, 7])
        n = max([maxy1 - miny1 + 1, maxy2 - miny2 + 1, 7])
        image1 = np.zeros((m, n))
        image2 = np.zeros((m, n))
        image1[outlines_array1[:, 0], outlines_array1[:, 1]] = 1
        image2[outlines_array2[:, 0], outlines_array2[:, 1]] = 1

        # 计算相似度
        similarity = 0
        try:
            similarity = structural_similarity(image1, image2)
        except Exception as e:
            logger.warning(e)
        return similarity

    def write_to_json_file(self, filament_datas):
        os.makedirs(self.output_dir + "/json_data", exist_ok=True)
        for filament_data in filament_datas:
            name, data = filament_data
            with open(self.output_dir + "/json_data/{}.json".format(name), "w") as f:
                f.write(json.dumps(data) + "\n")
        return 0

    def track_filament(self, need_track_filament, track_filament, track_data_dict):
        # 开始追踪
        track_filament_dict_list = defaultdict(list)
        track_filament_dict = dict()

        for current_filament in need_track_filament:
            current_filament_id = current_filament.get('id')

            for next_filament in sorted(track_filament, key=lambda x: x.get("centroid")[0], reverse=True):  # 一个按条纹
                next_filament_id = next_filament.get("id")

                # 计算相邻的图片的暗条的相似度
                similarity = self.get_similarity(current_filament, next_filament)
                if similarity >= self.similarity_threshold:
                    track_filament_dict_list[current_filament_id].append([next_filament_id, next_filament, similarity])
                # else:
                #     logger.warning("{} {} 相似度为:{}, 被忽略".format(current_filament_id, next_filament_id, similarity))

            filament_similarity_list = track_filament_dict_list.get(current_filament_id)
            if filament_similarity_list is not None:
                # 取相似度最好的下一张图片暗纹的id
                filament_similarity_list = sorted(filament_similarity_list, key=lambda x: x[-1], reverse=True)
                track_filament_dict[current_filament_id] = filament_similarity_list[0][0]

            else:

                logger.info("{} 当前暗条，在下一个图片中找不到相似的暗条".format(current_filament_id))

        track_data_dict.update({v: k for k, v in track_filament_dict.items()})
        return 0

    @staticmethod
    def get_similarity(current_filament, next_filament):
        similarity = 0
        x1, y1 = current_filament.get("centroid")
        # 基本特征向量
        feature1 = np.array(
            [current_filament.get("area") / 10, current_filament.get("length"), current_filament.get("width")])

        x2, y2 = next_filament.get("centroid")
        # 基本特征向量
        feature2 = np.array([next_filament.get("area") / 10, next_filament.get("length"), next_filament.get("width")])

        # 下一张图片上的暗纹在当前照片右侧
        if abs(x2 - x1) <= 10 and abs(y1 - y2) <= 15:
            #  可以尝试下面的相似度
            # feature_similarity = self.cos_theta(feature1, feature2)
            # outlines_similarity = self.outlines_similarity(current_filament.get("outlines"),
            #                                                next_filament.get("outlines"))

            # similarity = feature_similarity * outlines_similarity
            # similarity = outlines_similarity
            # similarity = feature_similarity

            # 满足这个条件，强制判断相似
            similarity = 1

        return similarity

    @staticmethod
    def update_filament_datas(filament_datas, track_data_dict):
        for item in filament_datas:
            _, filaments = item
            for filament in filaments:
                if track_data_dict.get(filament.get("id")) is not None:
                    filament.update({"id": track_data_dict.get(filament.get("id"))})
        return 0

    @staticmethod
    def update_figure_datas(figure_datas, filament_datas):
        n = len(figure_datas)
        for i in range(n):
            name, (corrected, rects) = figure_datas[i]
            _, filaments = filament_datas[i]
            for rect, filament in zip(rects, filaments):
                rect[0] = filament.get("id")
        return 0

    @staticmethod
    def deal_need_track_filament_id(need_track_filament, track_data_dict):
        # 获取当前跟踪字典中最大的id号，从这个号之后开始编号
        current_max_id = 0
        if track_data_dict:
            current_max_id = int(sorted(track_data_dict.values())[-1])

        need_track_filament_tmp0 = list()
        need_track_filament_tmp1 = list()
        for item in need_track_filament:
            if track_data_dict.get(item.get("id")) is not None:
                # 更新id
                item.update({"id": track_data_dict.get(item.get("id"))})
                need_track_filament_tmp0.append(item)
            else:
                # 需要编号的暗条
                need_track_filament_tmp1.append(item)

        if need_track_filament_tmp1:
            # 按中心y排序，从右外左编号
            need_track_filament_tmp = sorted(need_track_filament_tmp1, key=lambda x: x.get("centroid")[1], reverse=True)
            for num, item in enumerate(need_track_filament_tmp):
                # 记录原id和新的id的map
                new_id = "%04d" % (num + 1 + current_max_id)
                track_data_dict[item.get("id")] = new_id

                # 编号
                item.update({"id": new_id})

        need_track_filament_tmp0.extend(need_track_filament_tmp1)

        return need_track_filament_tmp0

    def full_trace(self, filament_datas, track_data_dict):
        # 增量追踪
        data = [item[1] for item in filament_datas]
        for i in range(len(data) - 1):
            # 当前图片
            need_track_filament = self.deal_need_track_filament_id(data[i], track_data_dict)

            # 追踪的图片
            self.track_filament(need_track_filament, data[i + 1], track_data_dict)

    def start(self, **kwargs):
        # 多少个图片参与运行。如果没有这个参数，全部参与。
        image_nums = kwargs.get("image_nums") or np.inf

        # 获取暗条数据，画图数据
        filament_datas, figure_datas = Recognize().get_filament_datas(input_dir=self.input_dir,
                                                                      image_nums=image_nums + 1)

        # 获取追踪的字典
        track_data_dict = dict()
        self.full_trace(filament_datas, track_data_dict)

        # 更新暗条数据，暗条id
        self.update_filament_datas(filament_datas, track_data_dict)

        # 输出暗条数据
        if kwargs.get("is_write_to_json"):
            self.write_to_json_file(filament_datas)

        # 更新画图数据，和暗条数据id
        self.update_figure_datas(figure_datas, filament_datas)

        # 画图
        for num, figure_data in enumerate(figure_datas):
            self.show_figure_data(figure_data, **kwargs)

            if num + 1 >= image_nums:
                break
        return 0


if __name__ == '__main__':
    Track().start(
        image_nums=2,
        is_write_to_json=True,
        is_save_image=True,
        is_show_image=False)
