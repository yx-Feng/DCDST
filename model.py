import torch
import torch.nn as nn
import math
from torch.nn import Parameter,TransformerEncoder, TransformerEncoderLayer
import torch.nn.functional as F
from torch_geometric.nn import GATConv


# 图卷积
class GCN(nn.Module):
    def __init__(self, ninput, nhid, noutput, dropout):
        super(GCN, self).__init__()
        self.gcn = nn.ModuleList() # 初始化一个ModuleList来存储GCN层
        self.dropout = dropout # 设置dropout率
        self.leaky_relu = nn.LeakyReLU(0.2) # 定义LeakyReLU激活函数，负斜率为0.2
        # 构建网络的通道数列表，包括输入层、隐藏层和输出层
        channels = [ninput] + nhid + [noutput]
        # 遍历通道数列表，构建GCN层
        for i in range(len(channels) - 1):
            # 创建图卷积层，输入维度为channels[i]，输出维度为channels[i + 1]
            gcn_layer = GraphConvolution(channels[i], channels[i + 1])
            self.gcn.append(gcn_layer) # 将图卷积层添加到ModuleList中

    def forward(self, x, adj):
        # 遍历所有的GCN层（除了最后一层）
        for i in range(len(self.gcn) - 1):
            x = self.leaky_relu(self.gcn[i](x, adj)) # 对输入x进行图卷积操作，然后应用LeakyReLU激活函数
        # 在最后一层之前应用dropout, self.training 是一个布尔值，用于指示模型当前是否处于训练模式
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gcn[-1](x, adj) # 最后一层图卷积操作，不应用激活函数
        return x # 返回最终的输出

# 图卷积层，核心功能是实现图卷积操作
class GraphConvolution(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features    # 输入特征的维度
        self.out_features = out_features  # 输出特征的维度
        self.weight = Parameter(torch.FloatTensor(in_features, out_features)) # 初始化权重矩阵
        # 是否使用偏置项
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    # 使用均匀分布初始化权重和偏置
    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv) # 将权重初始化为 [-stdv, stdv] 之间的均匀分布
        # 如果存在偏置项，同样使用均匀分布初始化
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    # 输入节点特征矩阵，形状为 (num_nodes, in_features)
    # adj：图的邻接矩阵，形状为 (num_nodes, num_nodes)
    def forward(self, input, adj):
        support = torch.mm(input, self.weight) # 将输入特征矩阵与权重矩阵相乘，得到中间结果 
        output = torch.spmm(adj, support)      # 用稀疏矩阵乘法 torch.spmm 将邻接矩阵 adj 与 support 相乘，得到输出特征矩阵
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    # 返回图卷积层的字符串表示，例如：GraphConvolution (64 -> 32)，表示输入维度为 64，输出维度为 32。
    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'


# 基于正弦函数的时间到向量（Time-to-Vector）的映射
class SineActivation(nn.Module):
    def __init__(self, in_features, out_features):
        super(SineActivation, self).__init__()
        self.out_features = out_features # 输出特征的维度
        # 定义可学习的参数 w0 和 b0，形状为 (in_features, 1)
        self.w0 = nn.parameter.Parameter(torch.randn(in_features, 1))
        self.b0 = nn.parameter.Parameter(torch.randn(in_features, 1))
        # 定义可学习的参数 w 和 b，形状为 (in_features, out_features - 1)
        self.w = nn.parameter.Parameter(torch.randn(in_features, out_features - 1))
        self.b = nn.parameter.Parameter(torch.randn(in_features, out_features - 1))
        # 定义激活函数为 torch.sin
        self.f = torch.sin

    def forward(self, tau):
        # 调用 t2v 函数，计算时间到向量的映射
        return self.t2v(tau, self.out_features)

    def t2v(self, tau, out_features, arg=None):
        if arg:
            v1 = self.f(torch.matmul(tau, self.w) + self.b, arg)
        else:
            v1 = self.f(torch.matmul(tau, self.w) + self.b)
        v2 = torch.matmul(tau, self.w0) + self.b0
        return torch.cat([v1, v2], 1)


# 将输入的时间 x 映射为一个高维向量
class Time2Vec(nn.Module):
    def __init__(self, out_dim):
        super(Time2Vec, self).__init__()
        self.l1 = SineActivation(1, out_dim)

    def forward(self, x):
        x = self.l1(x)
        return x   

# 嵌入融合
class FuseEmbeddings(nn.Module):
    def __init__(self, embed_dim_1, embed_dim_2, embed_dim_3):
        super(FuseEmbeddings, self).__init__() # 初始化类，调用父类构造函数
        embed_dim = embed_dim_1 + embed_dim_2 + embed_dim_3
        self.fuse_embed = nn.Linear(embed_dim, embed_dim)
        self.leaky_relu = nn.LeakyReLU(0.2) # 定义LeakyReLU激活函数，负斜率为0.2

    def forward(self, embed_1, embed_2, embed_3):
        x = torch.cat((embed_1, embed_2, embed_3), dim=-1)
        x = self.fuse_embed(x)
        x = self.leaky_relu(x) # 对线性层输出应用LeakyReLU激活函数
        return x # 返回处理后的嵌入

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=500):
        # 初始化位置编码层，d_model是嵌入的维度，dropout是丢弃率，max_len是序列的最大长度
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout) # 定义dropout层
        pe = torch.zeros(max_len, d_model) # 创建位置编码矩阵，大小为(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  # 创建位置索引
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)) # 定义位置编码的衰减因子
        # 对位置编码的偶数列应用正弦函数
        pe[:, 0::2] = torch.sin(position * div_term)
        # 对位置编码的奇数列应用余弦函数
        pe[:, 1::2] = torch.cos(position * div_term)
        # 调整位置编码的形状，增加批次维度并进行转置
        pe = pe.unsqueeze(0).transpose(0, 1)
        # 注册位置编码为模型的缓冲区，不会参与梯度更新
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :] # 将输入x和位置编码相加
        return self.dropout(x)  # 应用dropout


class TransformerModel(nn.Module):
    def __init__(self, num_poi, embed_size, nhead, nhid, nlayers, dropout=0.5):
        super(TransformerModel, self).__init__() # 初始化Transformer模型
        self.model_type = 'Transformer'
        self.pos_encoder = PositionalEncoding(embed_size, dropout) # 定义位置编码层
        encoder_layers = TransformerEncoderLayer(embed_size, nhead, nhid, dropout) # 定义Transformer的单层Encoder
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers) # 定义Transformer的完整Encoder，包含多个层
        self.embed_size = embed_size # 定义嵌入大小
        self.decoder_poi = nn.Linear(embed_size, num_poi) # 定义解码器
        self.init_weights() # 初始化解码器的权重

    def generate_square_subsequent_mask(self, sz):
        # 生成一个用于自回归模型的掩码（保证当前位置不能看到后续的位置）
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def init_weights(self):
        initrange = 0.1
        self.decoder_poi.bias.data.zero_()  # 将bias初始化为0
        self.decoder_poi.weight.data.uniform_(-initrange, initrange)  # 将权重初始化为均匀分布

    def forward(self, src, src_mask):
        # 前向传播过程
        # src = src * math.sqrt(self.embed_size)  # 对输入进行缩放
        # src = self.pos_encoder(src)  # 添加位置编码
        # x = self.transformer_encoder(src, src_mask)  # 通过Transformer编码器
        # out_poi = self.decoder_poi(x)  # 通过解码器得到POI预测
        # return out_poi
        B, L, _ = src.size()

        # 1) 把 (B, L, D) -> (L, B, D)
        src = src.transpose(0, 1)

        # 2) 如果没有传 mask 或尺寸不对，就基于 L 重新生成
        if src_mask is None or src_mask.size(0) != L or src_mask.size(1) != L:
            src_mask = self.generate_square_subsequent_mask(L, device=src.device)

        # 3) 标准 Transformer 流水线
        src = src * math.sqrt(self.embed_size)
        src = self.pos_encoder(src)  # 位置编码期望 (L, B, D)
        output = self.transformer_encoder(src, src_mask)  # (L, B, D)
        output = output.transpose(0, 1)                   # 再转回 (B, L, D)
        out_poi = self.decoder_poi(output)
        return out_poi

class SpatioTemporalBias(nn.Module):
    def __init__(self, nhead, input_dim=1, hidden_dim=8):
        super().__init__()
        # 使用 MLP
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, nhead)
        )

    def forward(self, matrix):
        if matrix.dim() == 2:
            matrix = matrix.unsqueeze(0)
        # 强制转换为 float
        matrix = matrix.float()
        batch_size, seq_len_x, seq_len_y = matrix.shape
    
        flat_values = matrix.reshape(-1, 1)
        bias = self.mlp(flat_values)
        return bias.view(batch_size, seq_len_x, seq_len_y, -1)

# 给Transformer模型添加时空偏置
class SpatioTemporalTransformerLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        # 多头注意力机制
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        # 时空偏置参数
        self.time_bias = SpatioTemporalBias(nhead)
        self.dist_bias = SpatioTemporalBias(nhead)
        # 前馈网络
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model)
        ) 
        self.ffn_gate = SpatioTemporalBias(nhead=1, input_dim=1, hidden_dim=8)
        # 正则化层
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src, t_deltas, dists, src_mask=None):
        batch_size, seq_len, _ = src.shape
        # 1. 多头注意力计算
        q = self.q_proj(src).view(batch_size, seq_len, self.nhead, -1).transpose(1,2)  # (bs, nhead, L, d_k)
        k = self.k_proj(src).view(batch_size, seq_len, self.nhead, -1).transpose(1,2)
        v = self.v_proj(src).view(batch_size, seq_len, self.nhead, -1).transpose(1,2) 
        # 计算注意力分数
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / (self.d_model ** 0.5)  # (bs, nhead, L, L)
        # 2. 时空偏置注入
        time_bias = self.time_bias(t_deltas).permute(0, 3, 1, 2)   # (batch_size, nhead, seq_len, seq_len)
        dist_bias = self.dist_bias(dists).permute(0, 3, 1, 2)        # (batch_size, nhead, seq_len, seq_len)
        attn_bias = time_bias + dist_bias
        # 添加偏置到注意力分数
        attn_scores += attn_bias
        # 3. 掩码处理
        if src_mask is not None:
            attn_scores = attn_scores.masked_fill(src_mask == 0, -1e9)
        # 4. softmax归一化
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_output = torch.matmul(attn_weights, v)  # (bs, nhead, L, d_k)
        # 5. 合并多头结果
        attn_output = attn_output.transpose(1,2).contiguous().view(batch_size, seq_len, -1)
        # 6. 残差连接+正则化
        src = src + self.dropout(attn_output)
        src = self.norm1(src)
        # 7. 前馈网络
        ffn_output = self.ffn(src)
        src = src + self.dropout(ffn_output)
        src = self.norm2(src)    
        return src


# 增强的Transformer模型实现
class SpatioTemporalTransformer(nn.Module):
    def __init__(self, num_pois, d_model, nhead, nhid, nlayers, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        self.layers = nn.ModuleList([
            SpatioTemporalTransformerLayer(d_model, nhead, nhid, dropout) 
            for _ in range(nlayers)
        ])
        self.norm = nn.LayerNorm(d_model)
        # 添加输出层
        self.output_layer = nn.Linear(d_model, num_pois)  # 新增
        # 新增对抗扰动投影层
        self.perturb_proj = nn.Linear(128, 192)  # POI嵌入128维 → 192维
        
    def forward(self, x, t_deltas, dists, src_mask=None):
        # 先对输入x加入位置编码
        x = self.pos_encoder(x)
        # 不再对x加上 time_enc 和 space_enc
        for layer in self.layers:
            x = layer(x, t_deltas, dists, src_mask)
        x = self.norm(x)
        return self.output_layer(x)  # 输出形状: (batch, seq_len, num_pois)

    def generate_square_subsequent_mask(self, sz):
        # 生成一个用于自回归模型的掩码（保证当前位置不能看到后续的位置）
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask
    
    def get_sequence_emb(self, x):
        # 返回最后一层的CLS token或平均池化
        return x.mean(dim=1)

# 对比损失实现
class NTXentLoss(nn.Module):
    def __init__(self, temperature=0.1):
        super().__init__()
        self.temperature = temperature
        self.cosine_sim = nn.CosineSimilarity(dim=-1)

    def forward(self, z1, z2):
        # 如果输入是序列或更高维度（比如 [B, T, D]），先沿时间维做平均池化
        if z1.dim() > 2:
            # 假设 z1/z2 形状是 [B, T, D] 或 [B, T, ... , D]，取 dim=1
            z1 = z1.mean(dim=1)
            z2 = z2.mean(dim=1)
        batch_size = z1.size(0)   # 现在 z1, z2 都是 [B, D]
        z = torch.cat([z1, z2], dim=0)  # [2B, D]
        # 1) 计算余弦相似度矩阵并除以温度
        sim = self.cosine_sim(z.unsqueeze(1), z.unsqueeze(0))  # [2B, 2B]
        sim = sim / self.temperature
        # 2) 排除对角线（自身对比）
        mask = torch.eye(2 * batch_size, dtype=torch.bool, device=z.device)
        sim = sim.masked_fill(mask, -float('inf'))
        # 3) 正例索引：前 B 个样本的正例是 i+B，后 B 个是 i-B
        idx = torch.arange(2 * batch_size, device=z.device)
        pos_idx = (idx + batch_size) % (2 * batch_size)
        # 4) 提取正例相似度，并从 sim 中屏蔽它
        positives = sim[idx, pos_idx].unsqueeze(1)             # [2B, 1]
        sim[idx, pos_idx] = -float('inf')
        # 5) 剩下都是负例
        negatives = sim                                       # [2B, 2B]
        # 6) 拼接 logits：正例在第 0 列，其它为负例
        logits = torch.cat([positives, negatives], dim=1)     # [2B, 1 + 2B]
        labels = torch.zeros(2 * batch_size, dtype=torch.long, device=z.device)  # [2B], 全 0
        # 7) 计算交叉熵损失
        loss = F.cross_entropy(logits, labels, reduction='mean')
        return loss
