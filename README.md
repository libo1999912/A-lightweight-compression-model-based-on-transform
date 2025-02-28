# 轻量化模型压缩与蒸馏
本项目旨在研究和实现轻量化模型压缩技术，主要通过知识蒸馏和剪枝技术来减少模型参数量，提高模型的推理速度和效率。项目中使用了Swin Transformer V2作为教师模型，并通过知识蒸馏训练一个较小的学生模型。然后对学生模型进行剪枝，并在剪枝后进行微调，以确保模型性能。

# 模型部署位置
GPU机器:/home/libo/libo/lightweight_model

# 文件结构
    lightweight_model.py: 主程序文件，包含模型训练、剪枝、微调和评估的主要逻辑。
    utils.py: 工具函数文件，包含数据处理、可视化和其他辅助函数。
    predictor.py: 预测和评估函数文件，包含计算RMSE、POD、FAR、CSI等指标的函数。
    dataset.py: 数据集处理文件，定义了数据加载和预处理的类。
    model/swin_transformer.py: Swin Transformer V2模型定义文件。

# 依赖

    Python 3.8+
    PyTorch 1.8+
    NumPy
    Matplotlib
    h5py
    loguru
    pandas
    torchvision
    adjustText


# conda环境
conda activate libo_py39

# 数据集  该项目使用的输入数据为雷达数据，存储在 .h5 格式的文件中。输入数据和标签包含雷达图像序列，经过预处理后供模型训练和测试。
    train_period_name: 训练数据集文件
    valid_period_name: 验证数据集文件
    test_period_name: 测试数据集文件

# 模型
本项目使用 Swin Transformer V2 作为教师模型和学生模型。教师模型为一个较大的网络，提供了较高的性能；学生模型则通过知识蒸馏学习教师模型的知识，并通过减小网络规模来提高其计算效率。
 
  教师模型
    使用较大的 Swin Transformer V2 模型，具有更高的性能。
    网络架构包括多个变换器层、头和窗口大小等，适用于大规模数据处理。

  学生模型
    学生模型是教师模型的简化版，采用更小的嵌入维度和较少的变换器层，以减少模型大小。
    学生模型通过蒸馏从教师模型中学习。

# 主要功能

  知识蒸馏
    知识蒸馏通过将教师模型的输出作为学生模型的指导信号，帮助学生模型在较小的架构下达到接近教师模型的表现。本项目中的蒸馏损失包括：
    软损失（Soft Loss）：基于教师和学生模型输出的相似度，使用KL散度来度量。
    硬损失（Hard Loss）：学生模型的输出与教师模型的输出之间的均方误差（MSE）。


  模型剪枝
    剪枝技术通过删除神经网络中不重要的参数（例如低权重的连接）来减小模型的体积，提高推理速度。我们在训练过程中每个 epoch 后对学生模型进行剪枝，剪去一定比例的参数。
    在此项目中，剪枝操作对卷积层和全连接层的权重进行 L1 范数剪枝。
    剪枝后，模型参数量会显著减少，并保持较好的预测精度。


# 训练过程
训练过程中，教师模型的参数被冻结，学生模型通过蒸馏损失进行优化。每个 epoch 后会进行剪枝，并打印出剪枝后的模型参数和训练损失。


# 预测结果保存
训练完毕后，会对验证集进行预测，并将预测结果保存为图像文件，便于后续分析。


# 启动训练
执行以下命令来启动训练：
python lightweight_model.py

# 运行结果示例
predicted_image_epoch_200:
![predicted_image_epoch_200](https://github.com/user-attachments/assets/02cfeb73-6461-472c-8b83-fa27699746de)

metrics_plot_epoch_200:
![metrics_plot_epoch_200](https://github.com/user-attachments/assets/2b32a883-2f89-484f-a2e7-636e07374e66)

rmse_plot_epoch_200:
![rmse_plot_epoch_200](https://github.com/user-attachments/assets/4118e220-b727-4e0f-9d9b-505e40913f4c)


