import numpy as np
import torch
import torch.nn.functional as F
from typing import List
device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
# 定义计算 RMSE 的函数
def calculate_rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())

# 提取预测步骤到一个单独的函数
def get_predictions(model, data_loader):
    model.eval()  # 切换到评估模式
    predictions_list = []
    labels_list = []

    with torch.no_grad():
        for input_, label in data_loader:
            input_, label = input_.float(), label.float()
            predicted_output = model(input_)
            predictions_list.append(predicted_output)
            labels_list.append(label)

    return predictions_list, labels_list

def prep_clf(obs: np.ndarray , sim: np.ndarray, grade_list: List=None,
             compare: str='>=', axis=None, return_array: bool=False):
    '''

    :param obs:
    :param sim:
    :param threshold:
    :param compare:
    :param axis:
    :return:
    '''

    if compare not in [">=",">","<","<="]:
        print("compare 参数只能是 >=   >  <  <=  中的一种")
        return
    if obs.shape != sim.shape:
        print('预报数据和观测数据维度不匹配')
        return

    if grade_list is None:
        grade_list = [1e-30]

    #Ob_shape = [obs[0,:,:].shape if axis==0 else (1,) ][0] # axis=0 == (lat,lon) | axis=(1,2) == t | axis=None == (1)
    Ob_shape = [obs.shape[-2:] if axis==0 else (1,) ][0] # axis=0 == (lat,lon) | axis=(1,2) == t | axis=None == (1)

    hfmc_array = np.zeros((len(grade_list), 4, *Ob_shape))
    print('>>>> (threshold, 混淆矩阵, lat, lon) = ',hfmc_array.shape)
    for i in range(len(grade_list)):
        threshold = grade_list[i]
        if compare == ">=":
            obs = np.where(obs >= threshold, 1, 0)
            sim = np.where(sim >= threshold, 1, 0)
        elif compare == "<=":
            obs = np.where(obs <= threshold, 1, 0)
            sim = np.where(sim <= threshold, 1, 0)
        elif compare == ">":
            obs = np.where(obs > threshold, 1, 0)
            sim = np.where(sim > threshold, 1, 0)
        elif compare == "<":
            obs = np.where(obs < threshold, 1, 0)
            sim = np.where(sim < threshold, 1, 0)

        # True positive (TP)
        '''
        hits = np.sum((obs == 1) & (sim == 1), axis = axis)
        # False negative (FN)
        misses = np.sum((obs == 1) & (sim == 0), axis = axis)
        # False positive (FP)
        falsealarms = np.sum((obs == 0) & (sim == 1), axis = axis)
        # True negative (TN)
        correctnegatives = np.sum((obs == 0) & (sim == 0), axis = axis)
        '''
        hits = np.where((obs == 1) & (sim == 1), 1, 0)
        # False negative (FN)
        misses = np.where((obs == 1) & (sim == 0), 1, 0)
        # False positive (FP)
        falsealarms = np.where((obs == 0) & (sim == 1), 1, 0)
        # True negative (TN)
        correctnegatives = np.where((obs == 0) & (sim == 0), 1, 0)

        if not return_array:
            hits = np.sum(hits, axis=axis)
            misses = np.sum(misses, axis=axis)
            falsealarms = np.sum(falsealarms, axis=axis)
            correctnegatives = np.sum(correctnegatives, axis=axis)

        # hfmc_array.append(hits)
        # hfmc_array.append(misses)
        # hfmc_array.append(falsealarms)
        # hfmc_array.append(correctnegatives)
        hfmc_array[i, 0, :] = hits
        hfmc_array[i, 1, :] = misses
        hfmc_array[i, 2, :] = falsealarms
        hfmc_array[i, 3, :] = correctnegatives

    Hits, Misses, Falsealarms, Correctnegatives =  hfmc_array[:,0, :], hfmc_array[:,1, :], hfmc_array[:,2, :], hfmc_array[:,3, :]

    #return Hits, Misses, Falsealarms, Correctnegatives
    res = np.array([Hits, Misses, Falsealarms, Correctnegatives])
    return res

# 在训练完成后计算评分指标
def compute_score_list(model, data_loader):
    time_steps = 10  # 假设时间步数为10
    rmse_lists = [[] for _ in range(time_steps)]  # 初始化每个时间步的RMSE列表

    predictions_list, labels_list = get_predictions(model, data_loader)
    #这是新加的
    # 如果 predicted_output 是元组，提取其中的张量
    if isinstance(predicted_output, tuple):
        predicted_output = predicted_output[0]
    # 如果 label 是元组，提取其中的张量
    if isinstance(label, tuple):
        label = label[0]
    # 添加调试信息
    print(f"predicted_output shape: {predicted_output.shape}")
    print(f"label shape: {label.shape}")

    # 确保 predicted_output 和 label 的形状一致
    if predicted_output.size() != label.size():
        label = label.view(predicted_output.size())
    ###到这里为止
    for predicted_output, label in zip(predictions_list, labels_list):
        for t in range(time_steps):
            rmse = calculate_rmse(predicted_output[:, t, :, :, :], label[:, t, :, :, :])
            rmse_lists[t].append(rmse)

    score_list = [np.mean(rmse_list) for rmse_list in rmse_lists]
    return score_list

# Calculate RMSE difference and check if within 10%
def calculate_rmse_difference(student_model, teacher_model, data_loader):
    student_rmse = compute_score_list(student_model, data_loader)
    teacher_rmse = compute_score_list(teacher_model, data_loader)

    rmse_diff = [abs(s - t) for s, t in zip(student_rmse, teacher_rmse)]  #计算两者的RMSE差值。
    avg_rmse_diff = np.mean(rmse_diff)  # 计算RMSE差值的平均值。
    avg_teacher_rmse = np.mean(teacher_rmse) # 计算教师模型的平均RMSE。
    # 计算RMSE差值占教师模型RMSE的百分比
    rmse_diff_percentage = (avg_rmse_diff / avg_teacher_rmse) * 100 # 计算RMSE差值占教师模型RMSE的百分比。
    rmse_diff_percentage = f"{rmse_diff_percentage:.2f}%"
    return rmse_diff_percentage



def compute_score_lists(model, data_loader, threshold=30):
    """
    计算给定模型和数据加载器的指标列表。

    参数:
        model: 要评估的模型。
        data_loader: 提供数据的数据加载器。
        threshold (float): 二分类的阈值。

    返回:
        list: 一个包含每个12分钟时间间隔计算出的指标的列表。
    """
    all_obs = []  # 用于存储所有标签的列表
    all_sim = []  # 用于存储所有模型输出的列表

    # 遍历数据加载器中的数据
    for data in data_loader:
        inputs, labels = data
        inputs = inputs.float().to(device)  # 将输入转换为float类型，并移动到设备（GPU/CPU）
        outputs = model(inputs)  # 使用模型进行预测

        # 将标签和模型输出转换为numpy数组并存储
        all_obs.append(labels.cpu().numpy())  # 将标签从GPU移动到CPU，并转换为numpy数组
        all_sim.append(outputs.detach().cpu().numpy())  # 将输出从GPU移动到CPU，并转换为numpy数组

    # 将所有批次的数据合并成一个大的数组
    obs = np.concatenate(all_obs, axis=0)  # 形状: (total_samples, num_intervals)
    sim = np.concatenate(all_sim, axis=0)  # 形状: (total_samples, num_intervals)

    # 去除多余的维度，确保它们是1D数组
    obs = np.squeeze(obs)  # 去除任何不必要的维度，确保为1D数组
    sim = np.squeeze(sim)  # 对sim执行相同的操作

    # 初始化存储每个时间间隔指标的列表
    pod_list = []
    far_list = []
    csi_list = []

    # 假设obs和sim的形状是 (total_samples, num_intervals)
    num_intervals = obs.shape[1]  # 时间间隔的数量是第二个维度

    # 遍历每个时间间隔并计算指标
    for i in range(num_intervals):
        # 对每个时间间隔计算混淆矩阵的组件
        hits, misses, falsealarms, correctnegatives = prep_clf(obs=obs[:, i]*70, sim=sim[:, i]*70, grade_list=[threshold],
                                                               compare='>=', axis=None, return_array=False)

        # 打印调试信息，查看hits、misses、falsealarms的值
        print(f"Interval {i}: hits={hits}, misses={misses}, falsealarms={falsealarms}")

        # 确保没有除零操作
        pod_score = hits / (hits + misses) if (hits + misses) != 0 else 0
        far_score = falsealarms / (hits + falsealarms) if (hits + falsealarms) != 0 else 0
        csi_score = hits / (hits + misses + falsealarms) if (hits + misses + falsealarms) != 0 else 0

        # 将计算出的指标追加到列表中
        pod_list.append(pod_score)
        far_list.append(far_score)
        csi_list.append(csi_score)

    # 确保返回的列表是扁平的1D数组（如果它们仍然是2D的）
    pod_list = np.array(pod_list).flatten()
    far_list = np.array(far_list).flatten()
    csi_list = np.array(csi_list).flatten()

    return [pod_list, far_list, csi_list]


def calculate_metrics_difference(student_model, teacher_model, data_loader):
    # 计算两个模型的指标
    student_score_lists = compute_score_lists(student_model, data_loader)
    teacher_score_lists = compute_score_lists(teacher_model, data_loader)

    # 初始化存储差值的列表
    pod_diff = []
    far_diff = []
    csi_diff = []

    # 计算每个时间步的差值
    for student_scores, teacher_scores in zip(student_score_lists, teacher_score_lists):
        pod_diff.append(abs(student_scores[0] - teacher_scores[0]))
        far_diff.append(abs(student_scores[1] - teacher_scores[1]))
        csi_diff.append(abs(student_scores[2] - teacher_scores[2]))

    # 计算平均差值
    avg_pod_diff = np.mean(pod_diff)
    avg_far_diff = np.mean(far_diff)
    avg_csi_diff = np.mean(csi_diff)
    # 计算教师模型的平均POD、FAR和CSI指标
    avg_teacher_pod = np.mean(teacher_score_lists[0])
    avg_teacher_far = np.mean(teacher_score_lists[1])
    avg_teacher_csi = np.mean(teacher_score_lists[2])
    # Calculate percentage differences relative to teacher model's average values
    pod_diff_percentage = (avg_pod_diff / avg_teacher_pod) * 100
    far_diff_percentage = (avg_far_diff / avg_teacher_far) * 100
    csi_diff_percentage = (avg_csi_diff / avg_teacher_csi) * 100
    # Format the percentage differences to two decimal places
    pod_diff_percentage = f"{pod_diff_percentage:.2f}%"
    far_diff_percentage = f"{far_diff_percentage:.2f}%"
    csi_diff_percentage = f"{csi_diff_percentage:.2f}%"
    return pod_diff_percentage, far_diff_percentage, csi_diff_percentage


