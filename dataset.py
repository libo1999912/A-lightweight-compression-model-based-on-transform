from abc import ABC, abstractmethod
import os
import sys
sys.path.append('..')

import h5py
import numpy as np
#import torch
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torchvision.transforms


#
class Data(ABC):
    """
        This class is the base class of Dataset

        Args:
            root_dir (str , optional): The root dir of input data

        Raise:
            TypeError: If the type of train_dir is not str.
            TypeError: If the type of test_dir is not str.
    """

    def __init__(self, data_params): # 初始化方法，设置数据集的根目录和训练、测试、验证数据的文件路径。
        self.root_dir = data_params.get('root_dir')
        self.train_name      =  os.path.join(self.root_dir, "train", data_params.get('train_period_name'))
        self.test_name       =  os.path.join(self.root_dir, "test", data_params.get('test_period_name'))
        self.valid_name      =  os.path.join(self.root_dir, "valid", data_params.get('valid_period_name'))

    @abstractmethod
    def __getitem__(self, index): # 抽象方法，需要在子类中实现，用于根据索引获取数据。
        """Defines behavior for when an item is accessed. Return the corresponding element for given index."""
        raise NotImplementedError(
            "{}.__getitem__ not implemented".format(self.dataset_type))

    @abstractmethod
    def __len__(self):  # 在子类中实现，用于返回数据集的长度。
        """Return length of dataset"""
        raise NotImplementedError(
            "{}.__len__ not implemented".format(self.dataset_type))


class RADARData(Data):
    """
    This class is used to processing radar dbz data. and is used generate the dataset generator .

    Args:

    Examples:
        >>>  data_params = \
        ...   { 'root_dir' : '.',
        ...     'radar_sta' : 'Z9010', # or 'ALL'
        ...   }
    """

    def __init__(self, data_params):
        super(RADARData, self).__init__(data_params)
        self.path       = data_params.get('root_dir')
        self.forecast_inputs     = data_params.get('forecast_inputs')
        self.forecast_steps      = data_params.get('forecast_steps')
        self.batch_size = data_params.get('batch_size')
        self._get_file(run_mode=data_params.get('run_mode')) # 加载数据

    def __len__(self):
        return self.radar_data.shape[0]

    def __getitem__(self, idx):

        data  = self.radar_data[idx, :self.forecast_inputs, : ,: ,: ]
        label = self.radar_data[idx,  self.forecast_inputs  : self.forecast_inputs + self.forecast_steps , : ,: ,: ]

        T, C, H, W = data.shape  # T 是时间步数，C 是通道数，H 和 W 是高度和宽度。
        T, C, H, W = label.shape
        yy, xx = np.ix_(np.arange(H), np.arange(W))
        circ = (2* (xx - xx.mean())/W)**2 + (2* (yy - yy.mean())/H)**2
        circ_idx = circ > 1
        data[:, :, circ_idx]  = 0
        label[:, :, circ_idx] = 0

        return data/70, label/70

    def _get_file(self, run_mode='train'):
        if run_mode == 'train':
            hf = h5py.File(self.train_name)
        elif run_mode == 'valid':
            hf = h5py.File(self.valid_name)
        elif run_mode == 'test':
            hf = h5py.File(self.test_name)

        self.radar_data = hf.get('radar_data')
        self.radar_data = self.radar_data[:, :, :, :, :]


if __name__ == '__main__':

    data_params = \
        {
        'root_dir'          : r'D:\PyCharm\meteorologyProject\轻量化模型压缩研究',
        'radar_sta'         : 'CHN',
        'run_mode'          : 'train',
        'train_period_name' : 'Radar_湖南_20240624_20240626_Input_5_Output_10_CHN.h5', # YYYY-MM-DD_YYYY-MM-DD_10_20_Radar_CHN_Sta.h5
        'valid_period_name' : 'Radar_Z9736_20240624_20240624_Input_5_Output_10_Z9736.h5',
        'test_period_name'  : 'Radar_湖南_20240624_20240626_Input_5_Output_10_CHN.h5',
        'batch_size'        : 8,
        'forecast_inputs'   : 5,
        'forecast_steps'    : 10

        }

    # 使用 torchvision.transforms.ToTensor() 将数据从 NumPy 数组转换为 PyTorch 的 tensor，并且将像素值归一化到 [0, 1]。
    # 使用 torchvision.transforms.Resize([160, 160]) 将图像的大小调整为 160x160
    data_transform = torchvision.transforms.Compose(
        [torchvision.transforms.ToTensor(),  # 将PIL Image 或者 ndarray 转成tensor 并且归一化到 0-1 之间，而且将[w,h,c] -> [c,w,h]
         torchvision.transforms.Resize([160, 160])
         ])

    label_transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),  # 将PIL Image 或者 ndarray 转成tensor 并且归一化到 0-1 之间，而且将[w,h,c] -> [c,w,h]
        torchvision.transforms.Resize([160, 160])
    ])
    train_set = RADARData(data_params) #创建train_set 实例，它是 RADARData 类的对象，负责加载和处理训练数据。

    train_data_loader = DataLoader(dataset = train_set, batch_size = train_set.batch_size) #, shuffle = train_set.shuffle, num_workers = train_set.num_workers)
    print(train_data_loader)
    # input_ 和 label 是数据和标签； 打印 input_.shape 和 label.shape，以确认数据的形状。
    for iteration , batch in enumerate(train_data_loader):
        input_, label = Variable(batch[0]),Variable(batch[1], requires_grad=False)
        print(input_.shape, label.shape)



