from pathlib import Path
import glob
import re
import numpy as np
import pandas as pd
from scipy.sparse.linalg import eigsh


# 生成一个保存目录，并确保目录名称唯一，例如 runs/exp --> runs/exp{sep}0, runs/exp{sep}1 ...
def increment_path(path, exist_ok=True, sep=''):
    path = Path(path)  # 转换为路径对象，确保跨平台兼容
    # 如果目录不存在或存在且允许重用，返回原路径
    if (path.exists() and exist_ok) or (not path.exists()):
        return str(path)
    else:
        dirs = glob.glob(f"{path}{sep}*")  # 查找所有类似的路径
        matches = [re.search(rf"%s{sep}(\d+)" % path.stem, d) for d in dirs] # 匹配类似的目录，提取数字部分
        i = [int(m.groups()[0]) for m in matches if m]  # 获取匹配的数字索引
        n = max(i) + 1 if i else 2  # 如果有匹配项，递增最大数字；否则从2开始
        return f"{path}{sep}{n}"  # 返回递增后的新目录路径

# 加载邻接矩阵A, 用于表示POI的邻接关系，维度=(node_num, node_num)
def load_graph_adj_matrix(path):
    A = np.loadtxt(path, delimiter=',')
    return A

# 加载距离矩阵D, 用于表示POI的距离关系，维度=(node_num, node_num)
def load_graph_dist_matrix(path):
    D = np.loadtxt(path, delimiter=',')
    return D

# 加载节点特征矩阵X，维度=(node_num, 4)
def load_graph_node_features(path, feature1, feature2, feature3):
    df = pd.read_csv(path)
    X = df[[feature1, feature2, feature3]].to_numpy()
    return X

# 计算拉普拉斯矩阵
def calculate_laplacian_matrix(adj_mat, mat_type):
    n_vertex = adj_mat.shape[0] # 图的结点数量
    # 行度矩阵
    deg_mat_row = np.asmatrix(np.diag(np.sum(adj_mat, axis=1)))
    # column sum
    # deg_mat_col = np.asmatrix(np.diag(np.sum(adj_mat, axis=0)))
    deg_mat = deg_mat_row

    adj_mat = np.asmatrix(adj_mat) # 将adj_mat转换为矩阵格式以便于矩阵运算
    id_mat = np.asmatrix(np.identity(n_vertex)) # 单位矩阵
    # 组合拉普拉斯矩阵（Combinatorial Laplacian Matrix）
    # L=D−A，其中𝐷是度矩阵, 𝐴是邻接矩阵
    if mat_type == 'com_lap_mat':
        com_lap_mat = deg_mat - adj_mat
        return com_lap_mat
    elif mat_type == 'wid_rw_normd_lap_mat':
        # 用于切比雪夫卷积（Chebyshev Convolution）的加宽归一化随机游走拉普拉斯矩阵
        rw_lap_mat = np.matmul(np.linalg.matrix_power(deg_mat, -1), adj_mat) # 计算随机游走矩阵D^(-1)*A
        rw_normd_lap_mat = id_mat - rw_lap_mat # 归一化随机游走矩阵
        lambda_max_rw = eigsh(rw_lap_mat, k=1, which='LM', return_eigenvectors=False)[0] # 使用特征值分解计算最大特征值，用于调整矩阵的范围
        wid_rw_normd_lap_mat = 2 * rw_normd_lap_mat / lambda_max_rw - id_mat # 加宽矩阵
        return wid_rw_normd_lap_mat
    elif mat_type == 'hat_rw_normd_lap_mat':
        # 用于GCN卷积的加帽归一化随机游走拉普拉斯矩阵
        wid_deg_mat = deg_mat + id_mat
        wid_adj_mat = adj_mat + id_mat
        hat_rw_normd_lap_mat = np.matmul(np.linalg.matrix_power(wid_deg_mat, -1), wid_adj_mat)
        return hat_rw_normd_lap_mat
    else:
        raise ValueError(f'ERROR: {mat_type} is unknown.')

# 计算带有掩码（mask）的均方误差（MSE）损失
def maksed_mse_loss(input, target, mask_value=-1):
    # 创建一个掩码，标记目标张量中值为mask_value的部分
    mask = target == mask_value
    # 计算输入和目标的平方差，忽略掩码部分
    out = (input[~mask] - target[~mask]) ** 2
    loss = out.mean() # 计算损失值的均值
    return loss  # 返回计算出的损失

# 计算 Top-K准确率
def top_k_acc_last_timestep(y_true_seq, y_pred_seq, k):
    y_true = y_true_seq[-1] # 真实的POI序列
    y_pred = y_pred_seq[-1] # 模型预测的POI序列
    top_k_rec = y_pred.argsort()[-k:][::-1] # 对预测POI进行排序，选取前K个预测POI
    
    # 如果真实的POI出现在前K个推荐的POI中，返回1，否则返回0
    idx = np.where(top_k_rec == y_true)[0]
    if len(idx) != 0:
        return 1
    else:
        return 0

# 计算平均精度 mAP = 1/rank+1, 即在前K个推荐的POI中，真实的下一个POI的排名位置越靠前，模型的性能越好
def mAP_metric_last_timestep(y_true_seq, y_pred_seq, k):
    # AP: area under PR curve
    # But in next POI rec, the number of positive sample is always 1. Precision is not well defined.
    # Take def of mAP from Personalized Long- and Short-term Preference Learning for Next POI Recommendation
    y_true = y_true_seq[-1]
    y_pred = y_pred_seq[-1]
    rec_list = y_pred.argsort()[-k:][::-1]
    r_idx = np.where(rec_list == y_true)[0]
    if len(r_idx) != 0:
        return 1 / (r_idx[0] + 1)
    else:
        return 0

# 计算均值倒数排名
def MRR_metric_last_timestep(y_true_seq, y_pred_seq):
    """ next poi metrics """
    # Mean Reciprocal Rank: Reciprocal of the rank of the first relevant item
    y_true = y_true_seq[-1]
    y_pred = y_pred_seq[-1]
    rec_list = y_pred.argsort()[-len(y_pred):][::-1]
    r_idx = np.where(rec_list == y_true)[0][0]
    return 1 / (r_idx + 1)

# 新增 NDCG 相关函数
def dcg_score_last_timestep(y_true_seq, y_pred_seq, k):
    """ next poi metrics: DCG score at the last timestep """
    y_true_id = y_true_seq[-1]  # 真实POI ID
    y_pred = y_pred_seq[-1]     # 预测分数向量

    # 生成理想one-hot向量
    num_poi = y_pred.shape[0]
    y_true_onehot = np.zeros(num_poi)
    if y_true_id < num_poi:  # 防止ID越界
        y_true_onehot[y_true_id] = 1

    # 计算预测排序的DCG
    order = np.argsort(y_pred)[::-1]  # 预测排序
    top_k_true = y_true_onehot[order[:k]]
    gains = 2 ** top_k_true - 1
    discounts = np.log2(np.arange(k) + 2)  # 修正discount计算
    return np.sum(gains / discounts)

def ndcg_score_last_timestep(y_true_seq, y_pred_seq, k):
    """ next poi metrics: NDCG score at the last timestep """
    # 生成理想DCG
    y_true_id = y_true_seq[-1]
    y_pred = y_pred_seq[-1]
    num_poi = y_pred.shape[0]

    ideal_true = np.zeros(num_poi)
    if y_true_id < num_poi:
        ideal_true[y_true_id] = 1

    # 计算理想排序的DCG（最佳情况）
    ideal_order = np.argsort(ideal_true)[::-1]
    ideal_gains = 2 ** ideal_true[ideal_order[:k]] - 1
    ideal_discounts = np.log2(np.arange(k) + 2)
    best_dcg = np.sum(ideal_gains / ideal_discounts)

    if best_dcg == 0:
        return 0.0

    # 计算当前DCG
    current_dcg = dcg_score_last_timestep(y_true_seq, y_pred_seq, k)
    return current_dcg / best_dcg

def coverage_at_ks_last_timestep(pred_pois):
    """
    pred_pois: (seq_len, num_poi) 的预测概率
    return: 字典 {k: 最后一个时间步的top-k POI索引集合}
    """
    last_step_pred = pred_pois[-1]  # 取最后一个时间步
    topk_indices = {
        k: set(np.argpartition(last_step_pred, -k)[-k:])
        for k in [5, 10, 20]
    }
    return topk_indices
