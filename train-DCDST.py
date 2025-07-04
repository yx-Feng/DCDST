from pathlib import Path
import os
import logging
# 1) 全局关闭 geohash 库的 INFO 日志
logging.getLogger("geohash").setLevel(logging.WARNING)
import yaml
import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import OneHotEncoder
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import torch.nn as nn
from tqdm import tqdm
import torch.optim as optim
import pygeohash as geohash
from scipy.spatial import cKDTree
from collections import defaultdict
from utils import *
from model import *
from param_parser import parameter_parser


def train(args):
    # 保存目录
    args.save_dir = increment_path(Path(args.project) / args.name, exist_ok=args.exist_ok, sep='-')
    if not os.path.exists(args.save_dir): os.makedirs(args.save_dir)
    # 日志
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(message)s', datefmt='%Y-%m-%d %H:%M:%S', filename=os.path.join(args.save_dir, f"log_training.txt"), filemode='w')
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)
    logging.getLogger('matplotlib.font_manager').disabled = True
    # 保存运行配置
    logging.info(args)
    with open(os.path.join(args.save_dir, 'args.yaml'), 'w') as f:
        yaml.dump(vars(args), f, sort_keys=False)

    # %% ====================== 加载数据 ======================
    # 训练集和验证集
    train_df = pd.read_csv(args.data_train)
    val_df = pd.read_csv(args.data_val)

    # 构建全局POI邻近图 (从train_df中构建)
    raw_D = load_graph_adj_matrix(args.data_dist_matrix)
    raw_X = load_graph_node_features(args.data_node_feats, args.feature1, args.feature2, args.feature3)
    num_pois = raw_X.shape[0]
    X = raw_X
    # Normalization
    D = calculate_laplacian_matrix(raw_D, mat_type='hat_rw_normd_lap_mat')

    # POI id映射到index
    nodes_df = pd.read_csv(args.data_node_feats)
    poi_ids = list(set(nodes_df['node_name/poi_id'].tolist()))
    poi_id2idx_dict = dict(zip(poi_ids, range(len(poi_ids))))
    # User id映射到index
    user_ids = [str(each) for each in list(set(train_df['user_id'].to_list()))]
    user_id2idx_dict = dict(zip(user_ids, range(len(user_ids))))

    # ====================== 数据集定义 ======================
    class TrajectoryDatasetTrain(Dataset):
        def __init__(self, train_df):
            self.df = train_df
            self.traj_seqs = []  # traj id: user id + traj no.
            self.input_seqs = []
            self.label_seqs = []
    
            for traj_id in tqdm(set(train_df['trajectory_id'].tolist())):
                traj_df = train_df[train_df['trajectory_id'] == traj_id]
                poi_ids = traj_df['POI_id'].to_list()
                poi_idxs = [poi_id2idx_dict[each] for each in poi_ids]
                time_feature = traj_df[args.time_feature].to_list()  # 提取时间特征
    
                input_seq = []
                label_seq = []
                for i in range(len(poi_idxs) - 1):
                    input_seq.append((poi_idxs[i], time_feature[i]))
                    label_seq.append((poi_idxs[i + 1]))
    
                if len(input_seq) < args.short_traj_thres:
                    continue
    
                self.traj_seqs.append(traj_id)
                self.input_seqs.append(input_seq)
                self.label_seqs.append(label_seq)
    
            # 基于Geohash的POI分组（每个POI有经纬度信息）
            self.poi_df = pd.read_csv(args.data_node_feats)
            self.geohash2pois = defaultdict(list)
            for idx, row in self.poi_df.iterrows():
                lat = row['latitude']
                lng = row['longitude']
                gh = geohash.encode(lat, lng, precision=6)  # 6位Geohash空间精度约±1.2km
                self.geohash2pois[gh].append(idx)
    
            # 高频POI替换池（按地理分组）
            self.high_freq_pois = [poi_id2idx_dict[p] for p in nodes_df.sort_values('checkin_cnt', ascending=False).head(1000)['node_name/poi_id']]
            self.high_freq_by_geohash = defaultdict(list)
            for p in self.high_freq_pois:
                lat = self.poi_df.iloc[p]['latitude']
                lng = self.poi_df.iloc[p]['longitude']
                gh = geohash.encode(lat, lng, precision=6)
                self.high_freq_by_geohash[gh].append(p)
    
            # 构建POI空间索引
            self.poi_coords = np.array([[row['latitude'], row['longitude']]  for _, row in self.poi_df.iterrows()])
            self.poi_kdtree = cKDTree(self.poi_coords)
            self.pop_threshold = self._calculate_popularity_threshold()
    
        def _get_spatiotemporal_matrix(self, poi_idxs, user_id, timestamps):
            """生成时空矩阵（时间差 + 哈弗辛距离）"""
            # 1) 将 datetime 转成 Unix 时间戳（秒）
            ts = np.array([pd.to_datetime(t).timestamp() for t in timestamps])
        
            # 2) 计算两两差值，单位小时
            t_deltas = np.abs(ts[:, None] - ts[None, :]) / 3600.0  # shape (L, L)
    
            # 3) 空间距离
            # 获取经纬度和小时信息（转换为弧度）
            lats = np.radians(self.poi_df.iloc[poi_idxs]['latitude'].values)
            lngs = np.radians(self.poi_df.iloc[poi_idxs]['longitude'].values)
            # Haversine距离计算
            lat1 = lats[:, np.newaxis]
            lat2 = lats[np.newaxis, :]
            dlat = lat1 - lat2
            lng1 = lngs[:, np.newaxis]
            lng2 = lngs[np.newaxis, :]
            dlng = lng1 - lng2
            a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlng / 2) ** 2
            c = 2 * np.arcsin(np.sqrt(a))
            earth_radius_km = 6371  # 地球平均半径，单位：km
            dists = earth_radius_km * c
            
            return t_deltas, dists

        def _calculate_popularity_threshold(self):
            """计算每个POI的替换概率阈值"""
            # 获取所有POI的访问次数
            checkin_counts = nodes_df['checkin_cnt'].values
            # 归一化为[0.3, 0.7]区间（示例值可调）
            min_cnt = np.min(checkin_counts)
            max_cnt = np.max(checkin_counts)
            thresholds = 0.3 + 0.4 * (checkin_counts - min_cnt) / (max_cnt - min_cnt)
            return torch.from_numpy(thresholds).float()

        def apply_group_augmentation(self, seq):
            """群体增强：替换高频POI"""
            aug_seq = []
            for (poi_idx, time_feat) in seq:
                if np.random.rand() < self.pop_threshold[poi_idx]:
                    lat = self.poi_df.iloc[poi_idx]['latitude']
                    lng = self.poi_df.iloc[poi_idx]['longitude']
                    gh = geohash.encode(lat, lng, precision=6)
                    candidates = self.high_freq_by_geohash.get(gh, [poi_idx])
                    new_poi_idx = np.random.choice(candidates)
                    aug_seq.append((new_poi_idx, time_feat)) # 保留时间特征
                else:
                    aug_seq.append((poi_idx, time_feat))
            return aug_seq

        def apply_individual_augmentation(self, seq):
            """个体增强：时空扰动"""
            perturbed_seq = []
            for (poi_idx, time_feat) in seq:
                if self.pop_threshold[poi_idx] < 0.3:  # 只对长尾POI增强
                    # 原有扰动逻辑
                    lat = self.poi_df.iloc[poi_idx]['latitude']
                    lng = self.poi_df.iloc[poi_idx]['longitude']
                    gh = geohash.encode(lat, lng, precision=7)
                    density = len(self.geohash2pois.get(gh, []))
                    max_shift = 0.01 / (1 + np.log1p(density))
                    
                    lat_perturb = lat + np.random.uniform(-max_shift, max_shift)
                    lng_perturb = lng + np.random.uniform(-max_shift, max_shift)
                    t_perturb = self.poi_df.iloc[poi_idx]['hour'] + np.random.uniform(-1, 1)
                    new_poi_idx = self.find_nearest_poi(lat_perturb, lng_perturb, t_perturb)
                    perturbed_seq.append((new_poi_idx, time_feat))
                else:
                    perturbed_seq.append((poi_idx, time_feat))
            return perturbed_seq
        
        def find_nearest_poi(self, lat, lng, t, time_tol=1):
            # 空间最近邻搜索
            dists, idxs = self.poi_kdtree.query([[lat, lng]], k=5)
            # 时间过滤
            valid = [i for i in idxs[0] if abs(self.poi_hours[i] - t) <= time_tol]
            return valid[0] if valid else idxs[0][0]
    
        def __len__(self):
            assert len(self.input_seqs) == len(self.label_seqs) == len(self.traj_seqs)
            return len(self.traj_seqs)
    
        def __getitem__(self, index):
            # 获取用户ID（假设在数据中有'user_id'字段）
            traj_id = self.traj_seqs[index]
            user_id = traj_id.split('_')[0]
            traj_df = self.df[self.df['trajectory_id'] == traj_id]
            # 原始序列
            input_seq = self.input_seqs[index]
            label_seq = self.label_seqs[index]
            # 并行生成两种增强
            group_aug_seq = self.apply_group_augmentation(input_seq)
            indiv_aug_seq = self.apply_individual_augmentation(input_seq)
            # 获取POI索引
            poi_idxs = [each[0] for each in input_seq]
            timestamps = traj_df['UTC_time'].tolist()[:len(input_seq)]
            # 获取时空矩阵
            t_deltas, dists = self._get_spatiotemporal_matrix(poi_idxs, user_id, timestamps)
            t_deltas = torch.from_numpy(t_deltas).to(device=args.device)
            dists = torch.from_numpy(dists).to(device=args.device)
            return (self.traj_seqs[index], input_seq, group_aug_seq, indiv_aug_seq, label_seq, t_deltas, dists, poi_idxs)

    class TrajectoryDatasetVal(Dataset):
        def __init__(self, df):
            self.df = df
            self.traj_seqs = []
            self.input_seqs = []
            self.label_seqs = []
    
            for traj_id in tqdm(set(df['trajectory_id'].tolist())):
                user_id = traj_id.split('_')[0]
                if user_id not in user_id2idx_dict.keys():
                    continue
                traj_df = df[df['trajectory_id'] == traj_id]
                poi_ids = traj_df['POI_id'].to_list()
                poi_idxs = []
                time_feature = traj_df[args.time_feature].to_list()
    
                for each in poi_ids:
                    if each in poi_id2idx_dict.keys():
                        poi_idxs.append(poi_id2idx_dict[each])
                    else:
                        continue
    
                input_seq = []
                label_seq = []
                for i in range(len(poi_idxs) - 1):
                    input_seq.append((poi_idxs[i], time_feature[i]))
                    label_seq.append((poi_idxs[i + 1]))
    
                if len(input_seq) < args.short_traj_thres:
                    continue
    
                self.input_seqs.append(input_seq)
                self.label_seqs.append(label_seq)
                self.traj_seqs.append(traj_id)
    
            self.poi_df = pd.read_csv(args.data_node_feats)  # 加载POI数据
            self.poi_coords = np.array([[row['latitude'], row['longitude']] for _, row in self.poi_df.iterrows()])
            self.poi_kdtree = cKDTree(self.poi_coords)
    
        def _get_spatiotemporal_matrix(self, poi_idxs, timestamps):
            """生成时空矩阵（时间差 + 哈弗辛距离）"""
            # 1) 将 datetime 转成 Unix 时间戳（秒）
            ts = np.array([pd.to_datetime(t).timestamp() for t in timestamps])
        
            # 2) 计算两两差值，单位小时
            t_deltas = np.abs(ts[:, None] - ts[None, :]) / 3600.0  # shape (L, L)
    
            # 3) 空间距离
            # 获取经纬度和小时信息（转换为弧度）
            lats = np.radians(self.poi_df.iloc[poi_idxs]['latitude'].values)
            lngs = np.radians(self.poi_df.iloc[poi_idxs]['longitude'].values)
            # Haversine距离计算
            lat1 = lats[:, np.newaxis]
            lat2 = lats[np.newaxis, :]
            dlat = lat1 - lat2
            lng1 = lngs[:, np.newaxis]
            lng2 = lngs[np.newaxis, :]
            dlng = lng1 - lng2
            a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlng / 2) ** 2
            c = 2 * np.arcsin(np.sqrt(a))
            earth_radius_km = 6371  # 地球平均半径，单位：km
            dists = earth_radius_km * c
            
            return t_deltas, dists
    
        def __len__(self):
            assert len(self.input_seqs) == len(self.label_seqs) == len(self.traj_seqs)
            return len(self.traj_seqs)
        
        def __getitem__(self, index):
            input_seq = self.input_seqs[index]
            label_seq = self.label_seqs[index]
            poi_idxs = [each[0] for each in input_seq] 
            # 获取用户 ID（假设在数据中有'user_id'字段）
            traj_id = self.traj_seqs[index]
            user_id = traj_id.split('_')[0]
            traj_df = self.df[self.df['trajectory_id'] == traj_id]
            timestamps = traj_df['UTC_time'].tolist()[:len(input_seq)]
            # 获取时空矩阵
            t_deltas, dists = self._get_spatiotemporal_matrix(poi_idxs, timestamps)
            t_deltas = torch.from_numpy(t_deltas).to(device=args.device)
            dists = torch.from_numpy(dists).to(device=args.device)
            return (self.traj_seqs[index], input_seq, input_seq, input_seq, label_seq, t_deltas, dists, poi_idxs)


    # %% ====================== Define dataloader ======================
    print('Prepare dataloader...')
    train_dataset = TrajectoryDatasetTrain(train_df)
    val_dataset = TrajectoryDatasetVal(val_df)
    train_loader = DataLoader(train_dataset, batch_size=args.batch, shuffle=True, drop_last=False, pin_memory=False, num_workers=args.workers, collate_fn=lambda x: x)
    val_loader = DataLoader(val_dataset, batch_size=args.batch, shuffle=False, drop_last=False, pin_memory=False, num_workers=args.workers, collate_fn=lambda x: x)

    # %% ====================== Build Models ======================
    # 嵌入模型
    if isinstance(X, np.ndarray):
        X = torch.from_numpy(X)
        D = torch.from_numpy(D)
    X = X.to(device=args.device, dtype=torch.float)
    D = D.to(device=args.device, dtype=torch.float)
    args.gcn_nfeat = X.shape[1]
    gcn_model = GCN(ninput=args.gcn_nfeat, nhid=args.gcn_nhid, noutput=args.dist_embed_dim, dropout=args.gcn_dropout)
    poi_embed_model = nn.Embedding(num_embeddings=num_pois, embedding_dim=args.poi_embed_dim)
    # 时间模型
    time_embed_model = Time2Vec(out_dim=args.time_embed_dim)
    # 嵌入融合模型
    embed_fuse_model = FuseEmbeddings(args.poi_embed_dim, args.time_embed_dim, args.dist_embed_dim)
    print(f"Time Embed Dim: {args.time_embed_dim}, Dist Embed Dim: {args.dist_embed_dim}")
    # 序列模型 128 + 32 + 32 = 192
    args.seq_input_embed = args.poi_embed_dim + args.time_embed_dim + args.dist_embed_dim
    #  ====================== 创新点1：时空偏置  ======================
    seq_model = SpatioTemporalTransformer(num_pois, d_model=args.seq_input_embed, nhead=args.transformer_nhead, nhid=args.transformer_nhid, nlayers=args.transformer_nlayers, dropout=args.transformer_dropout)
    # seq_model = TransformerModel(num_pois, args.seq_input_embed, args.transformer_nhead, args.transformer_nhid, args.transformer_nlayers, dropout=args.transformer_dropout)
    # 对比学习模块
    contrast_model = NTXentLoss(temperature=0.1)

    # 优化器
    optimizer = optim.Adam(params=list(poi_embed_model.parameters()) + list(time_embed_model.parameters()) + list(gcn_model.parameters()) 
            + list(embed_fuse_model.parameters()) + list(seq_model.parameters()) + list(contrast_model.parameters()), lr=args.lr, weight_decay=args.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', verbose=True, factor=args.lr_scheduler_factor)

    criterion_poi = nn.CrossEntropyLoss(ignore_index=-1)  # -1 is padding

    # %% Tool functions for training
    def input_traj_to_embeddings(sample, dist_embedding):
        traj_id = sample[0]
        input_seq = [each[0] for each in sample[1]]
        input_seq_time = [each[1] for each in sample[1]]
        # User to embedding
        user_id = traj_id.split('_')[0]
        user_idx = user_id2idx_dict[user_id]
        # POI to embedding and fuse embeddings
        input_seq_embed = []
        for idx in range(len(input_seq)):
            poi_idx = input_seq[idx]
            poi_idx_tensor = torch.LongTensor([poi_idx]).to(device=args.device)
            poi_embedding = poi_embed_model(poi_idx_tensor).squeeze(0)
            # 转换时间信息为嵌入
            time_embedding = time_embed_model(torch.tensor([input_seq_time[idx]], dtype=torch.float).to(device=args.device))
            time_embedding = torch.squeeze(time_embedding).to(device=args.device)
            current_dist_embedding = torch.squeeze(dist_embedding[poi_idx]).to(device=args.device)
            fused_embedding = embed_fuse_model(poi_embedding, time_embedding, current_dist_embedding)
            input_seq_embed.append(fused_embedding)
        return input_seq_embed

    # (B, L, L) 填充为 => (B, max_seq_len, max_seq_len)
    def pad_square_matrices(matrices, pad_value=0):
        max_size = max(mat.size(0) for mat in matrices)
        padded_matrices = []
        for mat in matrices:
            size = mat.size(0)
            if size < max_size:
                # pad: (left, right, top, bottom)，这里向右和下方填充
                pad = (0, max_size - size, 0, max_size - size)
                padded_matrices.append(F.pad(mat, pad, value=pad_value))
            else:
                padded_matrices.append(mat)
        return torch.stack(padded_matrices, dim=0)

    # 新增长尾POI识别函数
    def identify_tail_pois(nodes_df, poi_id2idx_dict, percentile=20):
        # 创建POI ID到checkin_cnt的映射
        poi_checkin = dict(zip(nodes_df['node_name/poi_id'], nodes_df['checkin_cnt']))
        # 按poi_id2idx_dict的顺序收集checkin_cnt
        checkin_list = []
        for poi_id in poi_id2idx_dict.keys():
            checkin_list.append(poi_checkin.get(poi_id, 0))  # 处理可能的缺失，假设默认为0
        checkin_counts = np.array(checkin_list)
        threshold = np.percentile(checkin_counts, percentile)
        tail_mask = checkin_counts < threshold
        return torch.where(torch.tensor(tail_mask))[0].tolist()
    # 在训练循环前初始化长尾POI列表
    tail_poi_indices = identify_tail_pois(nodes_df, poi_id2idx_dict, percentile=20)

    # 对抗训练函数（生成针对长尾POI嵌入的对抗扰动）
    def generate_tail_adversarial_noise(poi_embed_layer, tail_indices, loss, eps=0.2):
        if not tail_indices:
            return torch.zeros_like(poi_embed_layer.weight.data)
        # 正确获取可导的嵌入参数（非.data）
        weight = poi_embed_layer.weight  # 形状: (num_pois, embed_dim), requires_grad=True
        tail_embeddings = weight[tail_indices]  # 直接索引Parameter 
        # 必须重新设置requires_grad（确保梯度链路）
        tail_embeddings = tail_embeddings.requires_grad_(True)
        # 通过虚拟前向传播建立计算图关联
        dummy_loss = loss + 0.0 * tail_embeddings.sum()  # 强制关联 
        # 计算梯度
        grad = torch.autograd.grad(
            outputs=dummy_loss,
            inputs=tail_embeddings,
            retain_graph=True,
            create_graph=False,
            allow_unused=False  # 确保所有输入参与计算
        )[0]
        # 归一化并生成扰动
        norm = torch.norm(grad, p=2, dim=-1, keepdim=True)
        delta = eps * grad / (norm + 1e-8)
        # 构建全量扰动矩阵
        full_delta = torch.zeros_like(weight)
        full_delta[tail_indices] = delta.detach()  # 解除计算图
        return full_delta
    
    # %% ====================== Train ======================
    poi_embed_model = poi_embed_model.to(device=args.device)
    embed_fuse_model = embed_fuse_model.to(device=args.device)
    time_embed_model = time_embed_model.to(device=args.device)
    gcn_model = gcn_model.to(device=args.device)
    seq_model = seq_model.to(device=args.device)
    contrast_model = contrast_model.to(device=args.device)

    # %% Loop epoch
    epoch_metrics = {
        'train': defaultdict(list),
        'val': defaultdict(list)
    }

    # 初始化Early Stopping相关变量
    max_val_score = -np.inf
    early_stop_counter = 0

    # 新增动态门控参数
    w_alpha = nn.Parameter(torch.tensor(1.0))
    b_alpha = nn.Parameter(torch.tensor(-0.5))
    for epoch in range(args.epochs):
        logging.info(f"{'*' * 50}Epoch:{epoch:03d}{'*' * 50}\n")
        # ====================== 初始化覆盖集合 ======================
        train_covered_pois = {5: set(), 10: set(), 20: set()}  # 每个epoch初始化一次
        total_pois = len(poi_id2idx_dict)  # 总POI数
        models = [
             poi_embed_model, gcn_model, time_embed_model, embed_fuse_model, seq_model, contrast_model
        ]
        for model in models:
            model.train()

        train_batches_top1_acc_list, train_batches_top5_acc_list, train_batches_top10_acc_list, train_batches_top20_acc_list = [],[],[],[]
        train_batches_NDCG5_list, train_batches_NDCG10_list, train_batches_NDCG20_list = [],[],[]
        train_batches_mAP20_list, train_batches_mrr_list = [], []
        train_batches_loss_list, train_batches_poi_loss_list = [], []
        
        # Loop batch
        for b_idx, batch in enumerate(train_loader):
            batch_poi_indices = [sample[-1] for sample in batch]  # 获取每个样本的POI索引序列
            if len(batch) != args.batch:
                src_mask = seq_model.generate_square_subsequent_mask(len(batch)).to(args.device)

            # For padding
            batch_input_seqs, batch_seq_lens, orig_embeddings, group_embeddings,indiv_embeddings, batch_seq_labels_poi = [],[],[],[],[],[]
            t_delta_list, dists_list = [], []
            batch_poi_indices = []

            dist_embeddings = gcn_model(X, D)
            # Convert input seq to embeddings
            for sample in batch:
                traj_id, orig_seq, group_seq, indiv_seq, label_seq, t_deltas, dists, poi_idxs = sample 
                orig_emb = torch.stack(input_traj_to_embeddings((traj_id, orig_seq, label_seq), dist_embeddings)) # 原始序列嵌入
                group_emb = torch.stack(input_traj_to_embeddings((traj_id, group_seq, label_seq), dist_embeddings)) # 群体增强嵌入
                indiv_emb = torch.stack(input_traj_to_embeddings((traj_id, indiv_seq, label_seq), dist_embeddings)) # 个体增强嵌入   
                orig_embeddings.append(orig_emb)
                group_embeddings.append(group_emb)
                indiv_embeddings.append(indiv_emb)
                batch_input_seqs.append(orig_seq)
                batch_seq_lens.append(len(orig_seq))
                batch_seq_labels_poi.append(torch.LongTensor(label_seq))
                t_delta_list.append(t_deltas)  # 每个 t_deltas 的形状为 (L, L)
                dists_list.append(dists)       # 每个 dists 的形状为 (L, L)
                batch_poi_indices.append(poi_idxs)

            # 填充序列
            orig_padded = pad_sequence(orig_embeddings, batch_first=True, padding_value=-1)
            group_padded = pad_sequence(group_embeddings, batch_first=True, padding_value=-1)
            indiv_padded = pad_sequence(indiv_embeddings, batch_first=True, padding_value=-1)
            label_padded_poi = pad_sequence(batch_seq_labels_poi, batch_first=True, padding_value=-1)

            # 填充 t_deltas 和 dists（shape: (B, max_seq_len, max_seq_len)）
            padded_t_deltas = pad_square_matrices(t_delta_list, pad_value=0).to(device=args.device)
            padded_dists = pad_square_matrices(dists_list, pad_value=0).to(device=args.device)

            max_seq_len = orig_padded.shape[1]
            src_mask = seq_model.generate_square_subsequent_mask(max_seq_len).to(args.device)

            y_poi = label_padded_poi.to(device=args.device, dtype=torch.long)

            #  ====================== 创新点2：群体-个体解耦对比学习 ======================
            # 前向传播
            y_pred_poi = seq_model(orig_padded, padded_t_deltas, padded_dists, src_mask) # 原始序列
            group_output = seq_model(group_padded, padded_t_deltas, padded_dists, src_mask) # 群体增强
            indiv_output = seq_model(indiv_padded, padded_t_deltas, padded_dists, src_mask) # 个体增强
            # 验证形状
            # print("y_pred_poi.shape:", y_pred_poi.shape)
            # print("group_output.shape:", group_output.shape)
            # print("indiv_output.shape:", indiv_output.shape)
            # print("y_poi.shape:", y_poi.shape)
            # 计算对比损失
            L_group = contrast_model(y_pred_poi, group_output) # 群体对比
            L_indiv = contrast_model(y_pred_poi, indiv_output) # 个体对比
            # 动态门控加权
            popularity_scores = torch.tensor([
                np.mean([nodes_df.iloc[p[0]]['checkin_cnt'] for p in seq]) 
                for seq in batch_input_seqs
            ]).to(device=args.device) # 序列中POI的平均流行度
            C_u = popularity_scores.mean() # 用户从众度
            alpha = torch.sigmoid(w_alpha * C_u + b_alpha) # 动态权重
            L_contrast = alpha * L_group + (1 - alpha) * L_indiv
            
            loss_poi = criterion_poi(y_pred_poi.transpose(1, 2), y_poi)
            
            # 总损失
            # loss = loss_poi + args.contrast_weight * L_contrast + args.adv_weight * L_adv
            loss = loss_poi + args.contrast_weight * L_contrast
            
            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            optimizer.step()

            # Performance measurement
            top1_acc, top5_acc, top10_acc, top20_acc, ndcg5, ndcg10, ndcg20, mAP20, mrr = 0, 0, 0, 0, 0, 0, 0, 0, 0
            batch_label_pois = y_poi.detach().cpu().numpy()
            batch_pred_pois = y_pred_poi.detach().cpu().numpy()
            for label_pois, pred_pois, seq_len in zip(batch_label_pois, batch_pred_pois, batch_seq_lens):
                label_pois = label_pois[:seq_len]  # shape: (seq_len, )
                pred_pois = pred_pois[:seq_len, :]  # shape: (seq_len, num_poi)
                top1_acc += top_k_acc_last_timestep(label_pois, pred_pois, k=1)
                top5_acc += top_k_acc_last_timestep(label_pois, pred_pois, k=5)
                top10_acc += top_k_acc_last_timestep(label_pois, pred_pois, k=10)
                top20_acc += top_k_acc_last_timestep(label_pois, pred_pois, k=20)
                mAP20 += mAP_metric_last_timestep(label_pois, pred_pois, k=20)
                mrr += MRR_metric_last_timestep(label_pois, pred_pois)
                ndcg5 += ndcg_score_last_timestep(label_pois, pred_pois, k=5)
                ndcg10 += ndcg_score_last_timestep(label_pois, pred_pois, k=10)
                ndcg20 += ndcg_score_last_timestep(label_pois, pred_pois, k=20)
                # 获取当前样本的top-k推荐POI（最后一个时间步）
                coverage_dict = coverage_at_ks_last_timestep(pred_pois)
                # 将结果合并到全局覆盖集合
                for k in [5, 10, 20]:
                    train_covered_pois[k].update(coverage_dict[k])
                    
            train_batches_top1_acc_list.append(top1_acc / len(batch_label_pois))
            train_batches_top5_acc_list.append(top5_acc / len(batch_label_pois))
            train_batches_top10_acc_list.append(top10_acc / len(batch_label_pois))
            train_batches_top20_acc_list.append(top20_acc / len(batch_label_pois))
            train_batches_NDCG5_list.append(ndcg5 / len(batch_label_pois))
            train_batches_NDCG10_list.append(ndcg10 / len(batch_label_pois))
            train_batches_NDCG20_list.append(ndcg20 / len(batch_label_pois))
            train_batches_mAP20_list.append(mAP20 / len(batch_label_pois))
            train_batches_mrr_list.append(mrr / len(batch_label_pois))
            train_batches_loss_list.append(loss.detach().cpu().numpy())
            train_batches_poi_loss_list.append(loss_poi.detach().cpu().numpy())

            # Report training progress
            if (b_idx % (args.batch * 5)) == 0:
                sample_idx = 0
                batch_pred_pois_wo_attn = y_pred_poi.detach().cpu().numpy()
                logging.info(f'Epoch:{epoch}, batch:{b_idx}, train_batch_loss:{loss.item():.2f}, train_batch_top1_acc:{top1_acc / len(batch_label_pois):.2f}, '
                             f'train_move_loss:{np.mean(train_batches_loss_list):.2f}, train_move_poi_loss:{np.mean(train_batches_poi_loss_list):.2f}\n'
                             f'train_move_top1_acc:{np.mean(train_batches_top1_acc_list):.4f}, train_move_top5_acc:{np.mean(train_batches_top5_acc_list):.4f}, train_move_top10_acc:{np.mean(train_batches_top10_acc_list):.4f}, train_move_top20_acc:{np.mean(train_batches_top20_acc_list):.4f}\n'
                             f'train_move_mAP20:{np.mean(train_batches_mAP20_list):.4f}, train_move_MRR:{np.mean(train_batches_mrr_list):.4f}\n'
                             f'traj_id:{batch[sample_idx][0]}\n'
                             f'input_seq: {batch[sample_idx][1]}\n'
                             f'label_seq:{batch[sample_idx][2]}\n'
                             f'pred_seq_poi_wo_attn:{list(np.argmax(batch_pred_pois_wo_attn, axis=2)[sample_idx][:batch_seq_lens[sample_idx]])} \n'
                             f'pred_seq_poi:{list(np.argmax(batch_pred_pois, axis=2)[sample_idx][:batch_seq_lens[sample_idx]])} \n' + '=' * 100)

        # train end --------------------------------------------------------------------------------------------------------
        models = [
             poi_embed_model, gcn_model, time_embed_model, embed_fuse_model, seq_model, contrast_model
        ]
        for model in models:
            model.eval()
        
        # 初始化验证集的覆盖集合
        val_covered_pois = {5: set(), 10: set(), 20: set()}  # 每个 epoch 验证前初始化一次
        total_pois = len(poi_id2idx_dict)  # 总 POI 数（与训练集相同）
            
        val_batches_top1_acc_list, val_batches_top5_acc_list, val_batches_top10_acc_list, val_batches_top20_acc_list = [],[],[],[]
        val_batches_NDCG5_list, val_batches_NDCG10_list, val_batches_NDCG20_list = [], [], []
        val_batches_mAP20_list, val_batches_mrr_list = [], []
        val_batches_loss_list, val_batches_poi_loss_list = [], []
        for vb_idx, batch in enumerate(val_loader):
            if len(batch) != args.batch:
                src_mask = seq_model.generate_square_subsequent_mask(len(batch)).to(args.device)

            # For padding
            batch_input_seqs, batch_seq_lens, batch_seq_embeds, batch_seq_labels_poi = [],[],[],[]
            t_delta_list, dists_list = [], []
            batch_poi_indices = []

            dist_embeddings = gcn_model(X, D)
            # Convert input seq to embeddings
            for sample in batch:
                traj_id, orig_seq, _, _, label_seq, t_deltas, dists, poi_idxs = sample
                t_deltas_tensor = torch.tensor(t_deltas, dtype=torch.float32, device=args.device)
                dists_tensor = torch.tensor(dists, dtype=torch.float32, device=args.device)
                input_seq_embed = torch.stack(input_traj_to_embeddings((traj_id, orig_seq, label_seq), dist_embeddings))
                batch_seq_embeds.append(input_seq_embed)
                batch_seq_lens.append(len(orig_seq))
                batch_input_seqs.append(orig_seq)
                batch_seq_labels_poi.append(torch.LongTensor(label_seq))
                t_delta_list.append(t_deltas)  # 每个 t_deltas 的形状为 (L, L)
                dists_list.append(dists)       # 每个 dists 的形状为 (L, L)
                batch_poi_indices.append(poi_idxs)

            # Pad seqs for batch training
            batch_padded = pad_sequence(batch_seq_embeds, batch_first=True, padding_value=-1)
            label_padded_poi = pad_sequence(batch_seq_labels_poi, batch_first=True, padding_value=-1)

            # 填充 t_deltas 和 dists（shape: (B, max_seq_len, max_seq_len)）
            padded_t_deltas = pad_square_matrices(t_delta_list, pad_value=0).to(device=args.device)
            padded_dists = pad_square_matrices(dists_list, pad_value=0).to(device=args.device)

            max_seq_len = batch_padded.shape[1]
            src_mask = seq_model.generate_square_subsequent_mask(max_seq_len).to(args.device)
            
            # Feedforward
            x = batch_padded.to(device=args.device, dtype=torch.float)
            y_poi = label_padded_poi.to(device=args.device, dtype=torch.long)
            y_pred_poi = seq_model(x, padded_t_deltas, padded_dists, src_mask)

            # Calculate loss
            loss_poi = criterion_poi(y_pred_poi.transpose(1, 2), y_poi)
            loss = loss_poi
            
            # Performance measurement
            top1_acc, top5_acc, top10_acc, top20_acc, ndcg5, ndcg10, ndcg20, mAP20, mrr = 0, 0, 0, 0, 0, 0, 0, 0, 0
            batch_label_pois = y_poi.detach().cpu().numpy()
            batch_pred_pois = y_pred_poi.detach().cpu().numpy()
            for label_pois, pred_pois, seq_len in zip(batch_label_pois, batch_pred_pois, batch_seq_lens):
                label_pois = label_pois[:seq_len]  # shape: (seq_len, )
                pred_pois = pred_pois[:seq_len, :]  # shape: (seq_len, num_poi)
                top1_acc += top_k_acc_last_timestep(label_pois, pred_pois, k=1)
                top5_acc += top_k_acc_last_timestep(label_pois, pred_pois, k=5)
                top10_acc += top_k_acc_last_timestep(label_pois, pred_pois, k=10)
                top20_acc += top_k_acc_last_timestep(label_pois, pred_pois, k=20)
                mAP20 += mAP_metric_last_timestep(label_pois, pred_pois, k=20)
                mrr += MRR_metric_last_timestep(label_pois, pred_pois)
                ndcg5 += ndcg_score_last_timestep(label_pois, pred_pois, k=5)
                ndcg10 += ndcg_score_last_timestep(label_pois, pred_pois, k=10)
                ndcg20 += ndcg_score_last_timestep(label_pois, pred_pois, k=20)
                # 获取当前样本的 top-k 推荐 POI（最后一个时间步）
                coverage_dict = coverage_at_ks_last_timestep(pred_pois)
                # 合并到验证集覆盖集合
                for k in [5, 10, 20]:
                    val_covered_pois[k].update(coverage_dict[k])
                    
            val_batches_top1_acc_list.append(top1_acc / len(batch_label_pois))
            val_batches_top5_acc_list.append(top5_acc / len(batch_label_pois))
            val_batches_top10_acc_list.append(top10_acc / len(batch_label_pois))
            val_batches_top20_acc_list.append(top20_acc / len(batch_label_pois))
            val_batches_NDCG5_list.append(ndcg5 / len(batch_label_pois))
            val_batches_NDCG10_list.append(ndcg10 / len(batch_label_pois))
            val_batches_NDCG20_list.append(ndcg20 / len(batch_label_pois))
            val_batches_mAP20_list.append(mAP20 / len(batch_label_pois))
            val_batches_mrr_list.append(mrr / len(batch_label_pois))
            val_batches_loss_list.append(loss.detach().cpu().numpy())
            val_batches_poi_loss_list.append(loss_poi.detach().cpu().numpy())

            # Report validation progress
            if (vb_idx % (args.batch * 2)) == 0:
                sample_idx = 0
                batch_pred_pois_wo_attn = y_pred_poi.detach().cpu().numpy()
                logging.info(f'Epoch:{epoch}, batch:{vb_idx}, '
                             f'val_batch_loss:{loss.item():.2f}, val_batch_top1_acc:{top1_acc / len(batch_label_pois):.2f}, val_move_loss:{np.mean(val_batches_loss_list):.2f}, val_move_poi_loss:{np.mean(val_batches_poi_loss_list):.2f} \n'
                             f'val_move_top1_acc:{np.mean(val_batches_top1_acc_list):.4f}, val_move_top5_acc:{np.mean(val_batches_top5_acc_list):.4f}, val_move_top10_acc:{np.mean(val_batches_top10_acc_list):.4f}, val_move_top20_acc:{np.mean(val_batches_top20_acc_list):.4f} \n'
                             f'val_move_mAP20:{np.mean(val_batches_mAP20_list):.4f}, val_move_MRR:{np.mean(val_batches_mrr_list):.4f} \n'
                             f'traj_id:{batch[sample_idx][0]}\n'
                             f'input_seq:{batch[sample_idx][1]}\n'
                             f'label_seq:{batch[sample_idx][2]}\n'
                             f'pred_seq_poi_wo_attn:{list(np.argmax(batch_pred_pois_wo_attn, axis=2)[sample_idx][:batch_seq_lens[sample_idx]])} \n'
                             f'pred_seq_poi:{list(np.argmax(batch_pred_pois, axis=2)[sample_idx][:batch_seq_lens[sample_idx]])} \n' + '=' * 100)
        # valid end --------------------------------------------------------------------------------------------------------

        # 统计train和val的epoch指标
        metrics = [
            'top1_acc', 'top5_acc', 'top10_acc', 'top20_acc', 
            'mAP20', 'mrr', 'loss', 'poi_loss', 
            'NDCG5', 'NDCG10', 'NDCG20'
        ]
        
        batch_lists = {
            'train': {
                'top1_acc': train_batches_top1_acc_list, 'top5_acc': train_batches_top5_acc_list, 'top10_acc': train_batches_top10_acc_list, 'top20_acc': train_batches_top20_acc_list,
                'mAP20': train_batches_mAP20_list, 'mrr': train_batches_mrr_list, 'loss': train_batches_loss_list, 'poi_loss': train_batches_poi_loss_list,
                'NDCG5': train_batches_NDCG5_list, 'NDCG10': train_batches_NDCG10_list, 'NDCG20': train_batches_NDCG20_list,
            },
            'val': {
                'top1_acc': val_batches_top1_acc_list, 'top5_acc': val_batches_top5_acc_list, 'top10_acc': val_batches_top10_acc_list, 'top20_acc': val_batches_top20_acc_list,
                'mAP20': val_batches_mAP20_list, 'mrr': val_batches_mrr_list, 'loss': val_batches_loss_list, 'poi_loss': val_batches_poi_loss_list,
                'NDCG5': val_batches_NDCG5_list, 'NDCG10': val_batches_NDCG10_list, 'NDCG20': val_batches_NDCG20_list,
            }
        }
        
        for phase in ['train', 'val']:
            for metric in metrics:
                batch_list = batch_lists[phase][metric]
                epoch_value = np.mean(batch_list)
                epoch_metrics[phase][metric].append(epoch_value)

        # ====================== 计算整个epoch的覆盖率 ======================
        for k in [5, 10, 20]:
            coverage = len(train_covered_pois[k]) / total_pois
            epoch_metrics['train'][f'coverage@{k}'].append(coverage)
        # ====================== 计算验证集的全局覆盖率 ======================
        for k in [5, 10, 20]:
            coverage = len(val_covered_pois[k]) / total_pois
            epoch_metrics['val'][f'coverage@{k}'].append(coverage)

        # Monitor loss and score
        monitor_loss = epoch_metrics['val']['loss'][-1]
        monitor_score = 0.6 * epoch_metrics['val']['top5_acc'][-1] + 0.3 * epoch_metrics['val']['NDCG5'][-1] + 0.1 * epoch_metrics['val']['coverage@5'][-1]

        # Learning rate schuduler
        lr_scheduler.step(monitor_loss)

        # Print epoch results
        logging.info(
            f"Epoch {epoch}/{args.epochs}\n"
            f"train_loss:{epoch_metrics['train']['loss'][-1]:.4f}, train_poi_loss:{epoch_metrics['train']['poi_loss'][-1]:.4f}, "
            f"train_top1_acc:{epoch_metrics['train']['top1_acc'][-1]:.4f}, train_top5_acc:{epoch_metrics['train']['top5_acc'][-1]:.4f}, "
            f"train_top10_acc:{epoch_metrics['train']['top10_acc'][-1]:.4f}, train_top20_acc:{epoch_metrics['train']['top20_acc'][-1]:.4f}, "
            f"train_mAP20:{epoch_metrics['train']['mAP20'][-1]:.4f}, train_mrr:{epoch_metrics['train']['mrr'][-1]:.4f}, "
            f"train_NDCG5:{epoch_metrics['train']['NDCG5'][-1]:.4f}, train_NDCG10:{epoch_metrics['train']['NDCG10'][-1]:.4f}, train_NDCG20:{epoch_metrics['train']['NDCG20'][-1]:.4f}\n"
            f"val_loss:{epoch_metrics['val']['loss'][-1]:.4f}, val_poi_loss:{epoch_metrics['val']['poi_loss'][-1]:.4f}, "
            f"val_top1_acc:{epoch_metrics['val']['top1_acc'][-1]:.4f}, val_top5_acc:{epoch_metrics['val']['top5_acc'][-1]:.4f}, "
            f"val_top10_acc:{epoch_metrics['val']['top10_acc'][-1]:.4f}, val_top20_acc:{epoch_metrics['val']['top20_acc'][-1]:.4f}, "
            f"val_mAP20:{epoch_metrics['val']['mAP20'][-1]:.4f}, val_mrr:{epoch_metrics['val']['mrr'][-1]:.4f}, "
            f"val_NDCG5:{epoch_metrics['val']['NDCG5'][-1]:.4f}, val_NDCG10:{epoch_metrics['val']['NDCG10'][-1]:.4f}, val_NDCG20:{epoch_metrics['val']['NDCG20'][-1]:.4f}\n"
            f"train_coverage@5:{epoch_metrics['train']['coverage@5'][-1]:.4f}, "
            f"train_coverage@10:{epoch_metrics['train']['coverage@10'][-1]:.4f}, "
            f"train_coverage@20:{epoch_metrics['train']['coverage@20'][-1]:.4f}\n"
            f"val_coverage@5:{epoch_metrics['val']['coverage@5'][-1]:.4f}, "
            f"val_coverage@10:{epoch_metrics['val']['coverage@10'][-1]:.4f}, "
            f"val_coverage@20:{epoch_metrics['val']['coverage@20'][-1]:.4f}"
        )

        # Save model state dict
        if args.save_weights:
            state_dict = {
                'epoch': epoch,
                'poi_embed_state_dict': poi_embed_model.state_dict(),
                'embed_fuse_state_dict': embed_fuse_model.state_dict(),
                'seq_model_state_dict': seq_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'user_id2idx_dict': user_id2idx_dict,
                'poi_id2idx_dict': poi_id2idx_dict,
                'args': args,
                'epoch_train_metrics': {k: v[-1] for k, v in epoch_metrics['train'].items()},
                'epoch_val_metrics': {k: v[-1] for k, v in epoch_metrics['val'].items()}
            }
            model_save_dir = os.path.join(args.save_dir, 'checkpoints')
            save_model_flag = False
            # Save best val score epoch
            if monitor_score >= max_val_score:
                save_model_flag = True
                if not os.path.exists(model_save_dir): os.makedirs(model_save_dir)
                torch.save(state_dict, rf"{model_save_dir}/best_epoch.state.pt")
                with open(rf"{model_save_dir}/best_epoch.txt", 'w') as f:
                    print(state_dict['epoch_val_metrics'], file=f)
                max_val_score = monitor_score

        # 保存 train/val 指标
        with open(os.path.join(args.save_dir, 'metrics-train.txt'), "w") as f:
            for key, values in epoch_metrics['train'].items():
                formatted = [float(f"{v:.4f}") for v in values] # 格式化为4位小数
                print(f"train_epochs_{key}_list={formatted}", file=f)

        with open(os.path.join(args.save_dir, 'metrics-val.txt'), "w") as f:
            for key, values in epoch_metrics['val'].items():
                formatted = [float(f"{v:.4f}") for v in values]
                print(f"val_epochs_{key}_list={formatted}", file=f)

        # Early Stopping逻辑
        if save_model_flag:
            early_stop_counter = 0
        else:
            early_stop_counter += 1
            if early_stop_counter >= args.early_stop_patience:
                print(f"Early stopping triggered after {args.early_stop_patience} epochs without improvement.")
                break  # 终止训练循环

if __name__ == '__main__':
    args = parameter_parser()
    # The name of node features in graph_X.csv
    args.feature1 = 'checkin_cnt'
    args.feature2 = 'latitude'
    args.feature3 = 'longitude'
    train(args)
