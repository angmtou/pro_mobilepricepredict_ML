# -*- coding: utf-8 -*-

"""

    文件名:    config.py
    功能：     配置文件


"""
import os

# 指定数据集路径
dataset_path = './data'

# 结果保存路径
output_path = './output'
if not os.path.exists(output_path):
    os.makedirs(output_path)

# 数字数值列
numeric_cols = ['battery_power', 'clock_speed', 'fc', 'int_memory', 'm_dep', 'mobile_wt',
                'n_cores', 'pc', 'px_height', 'px_width', 'ram', 'sc_h', 'sc_w', 'talk_time']

# 类别数值列
cat_cols = ['blue', 'dual_sim', 'four_g', 'three_g', 'touch_screen', 'wifi']

# 使用的特征列
feat_cols = numeric_cols + cat_cols


# 标签列
label_col = 'price_range'
