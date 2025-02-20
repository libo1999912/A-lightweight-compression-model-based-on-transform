import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from dataset import RADARData
import torch.nn.utils.prune as prune
import matplotlib.pyplot as plt
import os
# from model import TransformerModel
from model.Basic_PredRNN_Seq2seq.PredRNN_Model import PredRNN
from utils import plt_metrics_lines, plt_metrics_test_lines, plt_radar_data_
from predictor import compute_score_list, compute_score_lists, get_predictions, calculate_rmse_difference, \
    calculate_metrics_difference

# 禁用 cuDNN
torch.backends.cudnn.enabled = False
device = torch.device('cuda:3' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')


# 知识蒸馏模型
class KnowledgeDistillationModel(nn.Module):
    def __init__(self, teacher_model, student_model):
        super(KnowledgeDistillationModel, self).__init__()
        self.teacher = teacher_model
        self.student = student_model

    def forward(self, x):
        teacher_output = self.teacher(x).to(device)  # 获取第一个元素并转移到设备  # 教师模型输出
        student_output = self.student(x).to(device)  # 获取第一个元素并转移到设备  # 学生模型输出
        # print("teacher_output:",teacher_output)
        # print("student_output:",student_output)
        return teacher_output, student_output


# 蒸馏损失函数
def distillation_loss(y_teacher, y_student, y_true, temperature=2.0, alpha=0.3):
    """
    蒸馏损失函数，适用于回归任务。
    :param y_teacher: 教师模型的预测值
    :param y_student: 学生模型的预测值
    :param y_true: 真实标签
    :param temperature: 用于软化输出的温度参数
    :param alpha: 软损失与硬损失的权重
    :return: 总损失
    """


    # 软损失：学生与教师模型预测的MSE损失，带温度平滑
    soft_loss = F.mse_loss(y_student / temperature, y_teacher_resized / temperature)
    # 硬损失：学生与真实标签的MSE损失
    hard_loss = F.mse_loss(y_student, y_true)
    print("y_teacher:", y_teacher.shape)
    print("y_student:", y_student.shape)
    print("y_true:", y_true.shape)
    # 总损失 = α * 软损失 + (1 - α) * 硬损失
    total_loss = alpha * soft_loss + (1 - alpha) * hard_loss
    return total_loss


# 剪枝函数
def prune_model(model, pruning_rate):
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) or isinstance(module, nn.Conv2d):
            prune.l1_unstructured(module, name='weight', amount=pruning_rate)
            prune.remove(module, 'weight')  # 移除剪枝后的权重


# 微调剪枝后的模型
def finetune_after_pruning(model, train_loader, optimizer, num_epochs=1, lr=1e-5):
    model.train()  # 设置为训练模式
    optimizer = optim.Adam(model.parameters(), lr=lr)  # 使用较低的学习率
    running_loss = 0.0

    for epoch in range(num_epochs):
        for batch_idx, (input_, label) in enumerate(train_loader):
            input_, label = input_.float().to(device), label.float().to(device)

            optimizer.zero_grad()  # 清空梯度

            # 正向传播
            teacher_output, student_output = model(input_)

            # 计算蒸馏损失
            distillation_loss_value = distillation_loss(teacher_output, student_output, label)

            # 计算分类损失
            # classification_loss = nn.CrossEntropyLoss()
            # classification_loss_value = classification_loss(student_output, label)
            # 调整 y_true 的形状与学生模型输出一致
            label = label.view_as(student_output)  # 使目标标签与学生模型的输出形状一致
            # 使用 MSE 损失代替分类损失
            classification_loss_value = F.mse_loss(student_output, label)
            # 总损失 = 蒸馏损失 + 分类损失
            distillation_alpha = 0.5  # 权重因子
            total_loss = distillation_alpha * distillation_loss_value + (
                    1 - distillation_alpha) * classification_loss_value

            # 反向传播并更新权重
            total_loss.backward()
            optimizer.step()

            running_loss += total_loss.item()

        # 打印当前epoch的损失
        print(f'Epoch [{epoch + 1}/{num_epochs}] Training Loss: {running_loss / len(train_loader)}')


# 检查剪枝效果
def check_pruned_parameters(model):
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) or isinstance(module, nn.Conv2d):
            # 打印剪枝后的非零权重数量
            non_zero_params = torch.sum(module.weight != 0).item()
            total_params = module.weight.numel()
            print(f"Layer {name}: Total params = {total_params}, Non-zero params = {non_zero_params}")


# 计算模型参数量
def calculate_model_size(model):
    return sum(p.numel() for p in model.parameters())


# 在训练结束后对某些数据进行预测并保存图像
def save_predictions(model, data_loader, output_dir="output_predictions", save_every_epoch=True):
    model.eval()  # 切换到评估模式
    os.makedirs(output_dir, exist_ok=True)  # 创建输出目录

    with torch.no_grad():
        for batch_idx, (input_, label) in enumerate(data_loader):
            input_ = input_.float().to(device)

            # 预测
            predicted_output = model(input_)

            # 只保存每个 epoch 之后的图像（如果设置了 save_every_epoch 为 True）
            if save_every_epoch and batch_idx == 0:  # 每个 epoch 第一个批次保存图像
                for i in range(predicted_output.size(0)):  # 遍历每个样本
                    # 获取最后时间步的预测图像 (10, 1, 256, 256)
                    final_pred_image = predicted_output[i, -1, 0, :, :].cpu().numpy()  # 获取最后一个时间步的 (256, 256)

                    # 保存图像
                    plt_radar_data_(input_[i].cpu().numpy(), final_pred_image, save_path=os.path.join(output_dir,
                                                                                                      f"predicted_image_epoch_{epoch}_batch_{batch_idx}_sample_{i}.png"))


def save_model(model, path):
    torch.save(model.state_dict(), path)


# 验证函数
def validate_model(model, valid_loader):
    model.eval()  # 切换到评估模式
    total_loss = 0.0
    total_samples = 0

    with torch.no_grad():  # 禁用梯度计算
        for input_, label in valid_loader:
            input_, label = input_.float().to(device), label.float().to(device)

            output = model(input_)
            # 使用 MSE loss 计算总损失
            loss = F.mse_loss(output, label, reduction='sum')  # reduction='sum' 会返回总损失
            total_loss += loss.item()
            total_samples += label.size(0)

    avg_loss = total_loss / total_samples  # 计算平均损失
    print(f'Validation Loss: {avg_loss:.4f}')


def train_and_prune_model(train_loader, valid_loader, distillation_model, optimizer, num_epochs, output_dir,
                          pruning_rate, eval_interval, pruning_epoch):
    """
    训练并剪枝模型的主函数
    :param train_loader: 训练数据加载器
    :param valid_loader: 验证数据加载器
    :param distillation_model: 包含 student 和 teacher 的蒸馏模型
    :param optimizer: 优化器
    :param num_epochs: 总训练轮数
    :param output_dir: 输出文件保存路径
    :param pruning_rate: 剪枝比例
    :param eval_interval: 评估间隔
    """
    os.makedirs(output_dir, exist_ok=True)
    # 设置剪枝的周期，假设每隔10个epoch进行一次剪枝
    # pruning_interval = 50  # 每50个epoch剪枝一次
    for epoch in range(1, num_epochs + 1):
        print(f"Epoch [{epoch}/{num_epochs}]")

        # 设置为训练模式
        distillation_model.train()
        running_loss = 0.0

        for batch_idx, (input_, label) in enumerate(train_loader):
            input_, label = input_.float().to(device), label.float().to(device)  # 转为浮点类型
            optimizer.zero_grad()  # 清空梯度

            # 前向传播
            teacher_output, student_output = distillation_model(input_)

            # 计算蒸馏损失（教师模型和学生模型输出之间的差异）
            distillation_loss_value = distillation_loss(teacher_output, student_output, label)

            # 计算分类损失（学生模型输出和真实标签之间的差异）
            # classification_loss = nn.CrossEntropyLoss()
            # classification_loss_value = classification_loss(student_output, label)
            # 调整 y_true 的形状与学生模型输出一致
            label = label.view_as(student_output)  # 使目标标签与学生模型的输出形状一致

            # 计算回归损失（MSE 损失）
            classification_loss_value = F.mse_loss(student_output, label)
            # 总损失 = 蒸馏损失 + 分类损失
            distillation_alpha = 0.5  # 权重因子
            total_loss = distillation_alpha * distillation_loss_value + (
                        1 - distillation_alpha) * classification_loss_value

            total_loss.backward()  # 反向传播
            optimizer.step()  # 更新参数

            running_loss += total_loss.item()

            # 每 10 个 batch 打印一次损失
            if batch_idx % 10 == 0:
                print(f'Epoch [{epoch}], Batch [{batch_idx}], Total Loss: {total_loss.item()}')

        # 第pruning_epoch轮进行剪枝
        if epoch == pruning_epoch:
            # 剪枝模型
            prune_model(distillation_model.student, pruning_rate=pruning_rate)
            check_pruned_parameters(distillation_model.student)
            student_params_after_pruning = calculate_model_size(distillation_model.student)
            print(f'Student model parameters after pruning: {student_params_after_pruning}')

            # 微调剪枝后的模型
            finetune_after_pruning(distillation_model, train_loader, optimizer, num_epochs=1, lr=1e-5)

        # 每隔 eval_interval 轮进行评估和可视化
        if epoch % eval_interval == 0:
            metrics_names = ['pod', 'far', 'csi']
            # score_list = compute_score_list(distillation_model.student, valid_loader)
            # plt_metrics_lines(score_list, forecast_time='2h', time_interval='12min', metrics_name='rmse',
            #                  save_path=os.path.join(output_dir, f'rmse_plot_epoch_{epoch}.png'))
            # score_lists = compute_score_lists(distillation_model.student, valid_loader)
            # plt_metrics_test_lines(score_lists, forecast_time='2h', time_interval='12min', metrics_names=metrics_names,
            #                       save_path=os.path.join(output_dir, f'metrics_plot_epoch_{epoch}.png'))
            student_score_list = compute_score_list(distillation_model.student, valid_loader)
            teacher_score_list = compute_score_list(distillation_model.teacher, valid_loader)
            plt_metrics_lines(student_score_list, teacher_score_list, forecast_time='2h', time_interval='12min',
                              metrics_name='rmse',
                              save_path=os.path.join(output_dir, f'rmse_plot_epoch_{epoch}.png'))
            # Example usage
            rmse_diff_percentage = calculate_rmse_difference(distillation_model.student, distillation_model.teacher,
                                                             valid_loader)
            print(f'Average RMSE difference: {rmse_diff_percentage}')
            student_score_lists = compute_score_lists(distillation_model.student, valid_loader)
            teacher_score_lists = compute_score_lists(distillation_model.teacher, valid_loader)
            plt_metrics_test_lines(student_score_lists, teacher_score_lists, forecast_time='2h', time_interval='12min',
                                   metrics_names=metrics_names,
                                   save_path=os.path.join(output_dir, f'metrics_plot_epoch_{epoch}.png'))
            pod_diff_percentage, far_diff_percentage, csi_diff_percentage = calculate_metrics_difference(
                distillation_model.student, distillation_model.teacher, valid_loader)
            print(f'Average POD difference: {pod_diff_percentage}')
            print(f'Average FAR difference: {far_diff_percentage}')
            print(f'Average CSI difference: {csi_diff_percentage}')
            # 保存预测图像
            predictions_list, labels_list = get_predictions(distillation_model.student, valid_loader)
            predictions_list_teacher, labels_list = get_predictions(distillation_model.teacher, valid_loader)
            plt_radar_data_(labels_list[0].cpu().numpy(), predictions_list[0].cpu().numpy(),
                            predictions_list_teacher[0].cpu().numpy(),
                            save_path=os.path.join(output_dir, f'predicted_image_epoch_{epoch}.png'))
            # 保存蒸馏剪枝完的学生模型
            save_model(student_model, os.path.join(output_dir, f'pruned_student_model_epoch_{epoch}.pth'))
            # 保存初始教师模型
            save_model(teacher_model, os.path.join(output_dir, f'teacher_model_epoch_{epoch}.pth'))
        # 调用验证函数
        validate_model(distillation_model.student, valid_loader)
        # 每个 epoch 后清理显存缓存
        torch.cuda.empty_cache()
        # 最后一次保存预测图像
    # predictions_list, labels_list = get_predictions(distillation_model.student, valid_loader)
    # plt_radar_data_(labels_list[0].cpu().numpy(), predictions_list[0].cpu().numpy(),
    #                save_path=os.path.join(output_dir, 'final_predicted_image.png'))


# 主程序
if __name__ == '__main__':
    data_params = {
        'root_dir': '/home/libo/libo/lightweight_model',
        'radar_sta': 'CHN',
        'run_mode': 'train',
        'train_period_name': 'Radar_湖南_20240624_20240626_Input_5_Output_10_CHN.h5',
        'valid_period_name': 'Radar_Z9736_20240624_20240624_Input_5_Output_10_Z9736.h5',
        'test_period_name': 'Radar_湖南_20200701_20200703_Input_5_Output_10_湖南.h5',
        'batch_size': 1,
        'forecast_inputs': 5,
        'forecast_steps': 10
    }
    data_params_valid = {
        'root_dir': '/home/libo/libo/lightweight_model',
        'radar_sta': 'CHN',
        'run_mode': 'valid',
        'train_period_name': 'Radar_湖南_20240624_20240626_Input_5_Output_10_CHN.h5',
        'valid_period_name': 'Radar_Z9736_20240624_20240624_Input_5_Output_10_Z9736.h5',
        'test_period_name': 'Radar_湖南_20200701_20200703_Input_5_Output_10_湖南.h5',
        'batch_size': 1,
        'forecast_inputs': 5,
        'forecast_steps': 10
    }

    # 数据加载
    train_set = RADARData(data_params)
    train_loader = DataLoader(dataset=train_set, batch_size=data_params['batch_size'], shuffle=True)

    valid_set = RADARData(data_params_valid)
    valid_loader = DataLoader(dataset=valid_set, batch_size=data_params_valid['batch_size'], shuffle=False)

    input_channels = data_params['forecast_inputs']  # 输入时间步数
    output_channels = data_params['forecast_steps']  # 输出时间步数
    seq_len = input_channels

    # 计算 feature_size (C * H * W)  即每个时间步的输入数据在经过展平（flatten）操作后的维度
    C, H, W = 1, 256, 256  # 根据你的数据形状，设置 C, H, W
    feature_size = C * H * W

    # 实例化教师模型
    teacher_model = PredRNN(input_size=(256, 256), # 输入图像的尺寸 (高度, 宽度)
                            input_dim=1,        # 输入图像的通道数，例如灰度图像为1，RGB图像为3
                            hidden_dim=[64, 64, 64, 64],  # 每一层的隐藏状态维度列表
                            hidden_dim_m=[64, 64, 64, 64],  # 每一层的记忆状态维度列表
                            kernel_size=(3, 3),         # 卷积核的尺寸 (高度, 宽度)
                            num_layers=4,                # 网络的层数
                            batch_first=True).to(device)  # 如果为True，则输入和输出的形状为 (batch_size, seq_len, ...)，否则为 (seq_len, batch_size, ...)

    # 实例化学生模型
    student_model = PredRNN(input_size=(256, 256),
                            input_dim=1,
                            hidden_dim=[64, 64],
                            hidden_dim_m=[64, 64],
                            kernel_size=(3, 3),
                            num_layers=2,
                            batch_first=True).to(device)
    # 冻结教师模型参数
    for param in teacher_model.parameters():  # 获取教师模型的所有参数（权重和偏置等）
        param.requires_grad = True  # 设置每个参数的 requires_grad 属性为 False，表示这些参数在反向传播时 不计算梯度，也不更新

    # 创建知识蒸馏模型
    distillation_model = KnowledgeDistillationModel(teacher_model, student_model).to(device)  # 将教师模型移到GPU

    # 打印学生模型和教师模型的参数量
    teacher_params = calculate_model_size(teacher_model)
    student_params = calculate_model_size(student_model)
    print(f'Teacher model parameters before pruning: {teacher_params}')
    print(f'Student model parameters before pruning: {student_params}')

    # 定义优化器   Adam优化器帮助在训练过程中更新模型参数，以最小化损失函数。
    # optimizer = optim.Adam(student_model.parameters(), lr=0.001)
    optimizer = optim.Adam(list(student_model.parameters()) + list(teacher_model.parameters()), lr=0.001)
    # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    # 调用训练并剪枝函数
    train_and_prune_model(
        train_loader=train_loader,  # 替换为实际的训练数据加载器
        valid_loader=valid_loader,  # 替换为实际的验证数据加载器
        distillation_model=distillation_model,  # 替换为实际的蒸馏模型
        optimizer=optimizer,  # 替换为实际的优化器
        num_epochs=200,
        output_dir='./output',
        pruning_rate=0.3,
        eval_interval=5,
        pruning_epoch=50
    )

    # # 打印剪枝后的参数
    student_params_after_pruning = calculate_model_size(student_model)
    print(f'Student model parameters after pruning: {student_params_after_pruning}')
    print(f'Compression ratio: {student_params_after_pruning / teacher_params:.2f}')
