#!usr/bin/env python
# -*- coding:utf-8 -*-
"""   
Author          Sun
Create          2022-04-10 6:49 PM
"""
import argparse
from app.track import Track

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_nums', type=int, default=5, required=False, help='参与运行图片数量，默认全体')
    parser.add_argument('--is_write_to_json', type=bool, default=True, required=False, help='是否存为json, 默认存')
    parser.add_argument('--is_save_image', type=bool, default=True, required=False, help='是否存图片，默认存')
    parser.add_argument('--is_show_image', type=bool, default=False, required=False, help='是否显示图片，默认不显示')

    args = parser.parse_args()
    Track().start(
        image_nums=args.image_nums,
        is_write_to_json=args.is_write_to_json,
        is_save_image=args.is_save_image,
        is_show_image=args.is_show_image
    )
