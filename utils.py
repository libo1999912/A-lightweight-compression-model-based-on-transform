#!/usr/bin/env python

# -*- coding:utf-8 -*-
"""
@author: zhangym
@contact: 976435584@qq.com
@software: PyCharm
@file: utils.py
@time: 2023/8/30 16:22
@Describe
@Version 1.0
"""
#import sys
# from data.makeData import ALL_ATMOSPHERIC_VARS
# from data.makeData import TARGET_SURFACE_VARS
# sys.path.append('../')

import matplotlib
from matplotlib.gridspec import GridSpec
import time
import torch
matplotlib.use('Agg')
#import matplotlib.pyplot as plt
#import matplotlib.font_manager as fm  #字体管理器
import numpy as np
from loguru import logger
import matplotlib.pyplot as plt
import os
import matplotlib as mpl
from adjustText import adjust_text
import pandas as pd

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'


log = logger.bind(module_name='Timer')


class Timer:
    def __init__(self, title):
        self.title = title
        self.start = time.time()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        elapsed = time.time() - self.start
        log.info(f"{self.title}  {elapsed} seconds.")

def make_dir(path) -> None:
    '''make_dir

    '''
    if os.path.exists(path):
        return
    try:
        permissions = os.R_OK | os.W_OK | os.X_OK
        os.umask(permissions << 3 | permissions)
        mode = permissions << 6
        os.makedirs(path, mode=mode, exist_ok=True)
    except PermissionError as e:
        raise TypeError("No write permission on the directory.") from e
    finally:
        pass


def get_normalize(root_dir, run_mode):
    '''
    :return mean = [14.93, 67.07, 823.51]
            std = [9.9, 23.51, 102.14]
    '''
    normalize_path = os.path.join(root_dir, run_mode)
    if not os.path.exists(normalize_path):
        raise FileExistsError(f'{normalize_path}')
    x_mean = np.load(os.path.join(normalize_path, 'x_mean.npy'))
    x_std = np.load(os.path.join(normalize_path, 'x_std.npy'))

    y_mean = np.load(os.path.join(normalize_path, 'y_mean.npy'))
    y_std = np.load(os.path.join(normalize_path, 'y_std.npy'))


    return x_mean, x_std, y_mean, y_std

def plt_radar_data(x, y):
    """
    Visualize the forecast results in T+30 , T+60, T+90 min.

    Args:
        x (Tensor): The groundtruth of precipitation in 90 min.
        y (int): The prediction of precipitation in 90 min.
    """
    fig_num = 3
    fig = plt.figure(figsize=(20, 8))
    tget = np.expand_dims(y.asnumpy().squeeze(0), axis=-1)
    for i in range(1, fig_num + 1):
        ax = fig.add_subplot(2, fig_num, i)
        ax.imshow(tget[5 + (i - 1) * 6, ..., 0], vmin=0, vmax=10, cmap="jet")
    pred = x.asnumpy().squeeze(0)
    for i in range(fig_num + 1, 2 * fig_num + 1):
        ax = fig.add_subplot(2, fig_num, i)
        ax.imshow(pred[4 + (i - fig_num - 1) * 6, ...], vmin=0, vmax=10, cmap="jet")
    plt.savefig(f'pred_result.png')
    plt.show()


def colormap():

    return mpl.colors.LinearSegmentedColormap.from_list('cmap', ['#01A0F6', '#00ECEC', '#00D800', '#019000', '#FFFF00',
                                                                 '#E7C000', '#FF9000', '#FF0000', '#D60000', '#C00000',
                                                                 '#FF00F0', '#9600B4', '#AD90F0'], 13)

def plt_radar_data_(x, y,teacher, save_path = 'test.png'):
    '''
     Visualize the forecast results in T+30 , T+60, T+90 min.

    :param x: (frame_n , w, h) , groundtruth
    :param y: (frame_n, w, h)  , precipitation
    :return:
    '''

    fig_num = 10  # 10 个时间步
    fig = plt.figure(figsize=(30, 8))
    fig.suptitle('Radar Title')
    # 创建 GridSpec 对象
    gs = GridSpec(3, fig_num, figure=fig)  # 3行，10列
    
    tget =  x.squeeze() * 70  #x.shape = (10, 256, 256)
    tget[tget <= 10] = 0
    for i in range(1, fig_num + 1):
        ax = fig.add_subplot(gs[0,  i -1])
        cax = ax.imshow(tget[i - 1, : , :], vmin=0, vmax=70, cmap=colormap())
        ax.set_title(f'-{i}-')

    pred =  y.squeeze() * 70
    pred[pred <= 10] = 0
    for i in range(fig_num + 1, 2 * fig_num + 1):
        ax = fig.add_subplot(gs[1,  i - fig_num -1])
        # ax.imshow(pred[i - fig_num - 1, :, :], vmin=0, vmax=70, cmap="jet")
        print('pred index:',i - fig_num - 1)
        cax = ax.imshow(pred[i - fig_num - 1, :, :], vmin=0, vmax=70, cmap=colormap())
        ax.set_title(f'-{i-fig_num}-')
    # 第三列：教师模型预测的雷达图
    teacher_pred = teacher.squeeze() * 70
    teacher_pred[teacher_pred <= 10] = 0
    for i in range(fig_num + 1, 2 * fig_num + 1):
        ax = fig.add_subplot(gs[2, i - fig_num - 1])
        # 直接索引数据，不需要在中间加入额外的维度
        cax = ax.imshow(teacher_pred[i - fig_num - 1, :, :], vmin=0, vmax=70, cmap=colormap())
        ax.set_title(f'-{i - fig_num}-')

    # 在右侧添加一个整体色标
    cbar_ax = fig.add_axes([0.92, 0.15, 0.01, 0.7])  # [left, bottom, width, height]
    fig.colorbar(cax, cax=cbar_ax)

    plt.subplots_adjust(left=0.05, right=0.90, top=0.85, bottom=0.10, wspace=0.4, hspace=0.4)
    plt.savefig(save_path)

# 用于绘制单条评分折线图，显示在不同时间间隔下的某个评分指标（如 RMSE）的变化。
def plt_metrics_lines(student_score_list,teacher_score_list,forecast_time = '2h', time_interval = '12min', metrics_name = 'rmse', save_path = '{epoch}_{rmse}.png'):
    ''' drop vaild lines plot, rmse and ssim and csi , 0 - 2h (12min)
        Args:
            score_list: 0-2h评分列表
            forecast_time（2h）: 预报时间，
            time_interval（12min）: 时间间隔
    '''
    forecast_minutes = int(forecast_time[:-1]) * 60 if forecast_time[-1] == 'h' else int(forecast_time[:-1])
    interval_minutes = int(time_interval[:-3]) if time_interval[-3:] == 'min' else int(time_interval[:-1]) * 60

    # 生成时间轴的刻度
    time_ticks = np.arange(12, forecast_minutes + interval_minutes, interval_minutes)
    time_labels = [f'{int(tick // 60)}h{int(tick % 60)}min' for tick in time_ticks]

    # 画出折线图
    plt.figure(figsize=(10, 8))
    #plt.plot(time_ticks, score_list, marker='o', linestyle='-', color='b', label='Student Model')
    plt.plot(time_ticks, student_score_list, marker='o', linestyle='-', color='b', label='Student Model')
    plt.plot(time_ticks, teacher_score_list, marker='o', linestyle='-', color='r', label='Teacher Model')
    # 设置x轴刻度和标签
    plt.xticks(time_ticks, time_labels, rotation=45)

    # 添加标题和标签
    #plt.title('Radar 0-2H Metrics ')
    plt.title(f'{metrics_name.upper()} over {forecast_time}')
    plt.xlabel('Time Interval')
    #plt.ylabel(metrics_name)
    plt.ylabel(metrics_name.upper())
    plt.legend()
    # 显示网格
    plt.grid(True)

    # # 保存图像
    # epoch = 1  # 这里你可以根据需要设置实际的epoch值
    # rmse = round(np.sqrt(np.mean(np.square(score_list))), 2)
    # save_filename = save_path.format(epoch=epoch, rmse=rmse)
    plt.savefig(save_path)
    # 显示图像
    # plt.show()

# 用于绘制多条评分折线图，显示在不同时间间隔下的多个评分指标（如 POD、FAR、CSI）的变化。
def plt_metrics_test_lines(student_score_lists, teacher_score_lists, forecast_time = '2h', time_interval = '12min', metrics_names = ['pod', 'far', 'csi'], save_path = '{epoch}_{rmse}.png'):
    '''

    :param score_lists: [[0.7, 0.65], [0.54, 0.34], [0.2, 0.14]]
    :param forecast_time:
    :param time_interval:
    :param metrics_names:
    :param save_path:
    :return:
    '''

    #assert len(score_lists) == len(metrics_names), 'score_lists 纬度不对 '
    assert len(student_score_lists) == len(
        metrics_names), 'student_score_lists length does not match metrics_names length'
    assert len(teacher_score_lists) == len(
        metrics_names), 'teacher_score_lists length does not match metrics_names length'

    metricsname = metrics_names[0].split('-')[0]
    forecast_minutes = int(forecast_time[:-1]) * 60 if forecast_time[-1] == 'h' else int(forecast_time[:-1])
    interval_minutes = int(time_interval[:-3]) if time_interval[-3:] == 'min' else int(time_interval[:-1]) * 60

    # 生成时间轴的刻度
    time_ticks = np.arange(12, forecast_minutes + interval_minutes, interval_minutes)
    time_labels = [f'{int(tick // 60)}h{int(tick % 60)}min' for tick in time_ticks]

    # 画出折线图
    plt.figure(figsize=(10, 8))
    colors = ['r', 'b', 'y', 'g']
    texts = []
    DATA  = []
    # for item in zip(score_lists, colors, metrics_names):
    #     plt.plot(time_ticks, item[0], marker='o', linestyle='-', color=item[1], label= item[2])
    #     plt.axhline(np.mean(item[0]), color=item[1], linestyle='--')
    #     texts.append(plt.text(5, np.mean(item[0]), np.mean(item[0]), fontsize=12, ha='right', va='bottom', color=item[1]))
    #     dataframe_dict = {'time': time_ticks, item[2]: item[0]}
    #     df = pd.DataFrame(dataframe_dict)
    #     DATA.append(df)
    #
    # df_con = pd.concat(DATA, axis=1)
    # df_con.to_csv(save_path.replace('.png', '.csv'))
    #
    # adjust_text(texts)
    for student_scores, teacher_scores, color, metric_name in zip(student_score_lists, teacher_score_lists, colors, metrics_names):
        plt.plot(time_ticks, student_scores, marker='o', linestyle='-', color=color, label=f'Student {metric_name}')
        plt.plot(time_ticks, teacher_scores, marker='x', linestyle='--', color=color, label=f'Teacher {metric_name}')
    # 设置x轴刻度和标签
    plt.xticks(time_ticks, time_labels, rotation=45)

    # 添加标题和标签
    plt.title('Radar 0-2H Metrics ')
    plt.xlabel('Time Interval')
    plt.ylabel(f'{metricsname} - score')

    # 手动调整 y 轴范围，增加上下边距
    # plt.ylim(5, 35)
    # 显示图例
    plt.legend()

    # 显示网格
    plt.grid(True)
    plt.savefig(save_path)



if __name__ == '__main__':
    score_list = [0.78, 0.75, 0.76, 0.68, 0.64, 0.62, 0.6, 0.58, 0.5, 0.3]
    plt_metrics_lines(score_list)


    import  h5py as hf
    f = hf.File('/share/HPC/home/zhaox_hx/data/aidata/zhangym_tmp/DATA/ML_dataset/Radar_datasets/valid/Radar_20210701_20210702_Input:5_Output:10_CHN.h5')
    print(f.keys())
    data = f.get('radar_data')
    index = 50
    origin_data, target_data = data[index , :  , :, : , :], data[index ,  : , : , : , :]
    print(origin_data.shape)
    with Timer('测试'):
        plt_radar_data_(x = origin_data / 255, y = target_data / 255)
