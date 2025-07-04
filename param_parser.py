import argparse
import torch

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')


def parameter_parser():
    parser = argparse.ArgumentParser(description="Run Model.")
    
    parser.add_argument('--device',type=str,default=device,help='')

    # 实验配置
    parser.add_argument('--project',default='runs/train',help='save to project/name')
    parser.add_argument('--name',default='exp',help='save to project/name')
    parser.add_argument('--exist-ok',action='store_true',help='existing project/name ok, do not increment')
    parser.add_argument('--workers',type=int, default=0,help='Num of workers for dataloader.')
    parser.add_argument('--save-embeds',action='store_true',default=False,help='whether save the embeddings')
    parser.add_argument('--save-weights',action='store_true',default=True, help='whether save the model')
    
    # 数据集
    parser.add_argument('--data-train',type=str,default='dataset/foursquare/nyc/nyc_train.csv',help='Training data path')
    parser.add_argument('--data-val', type=str, default='dataset/foursquare/nyc/nyc_val.csv', help='Validation data path')
    parser.add_argument('--data-adj-matrix',type=str,default='dataset/foursquare/nyc/graph_A.csv',help='Graph adjacent path')
    parser.add_argument('--data-dist-matrix',type=str,default='dataset/foursquare/nyc/graph_dist.csv',help='Graph dist path')
    parser.add_argument('--data-node-feats',type=str,default='dataset/foursquare/nyc/graph_X.csv',help='Graph node features path')
    parser.add_argument('--user-edges',type=str,default='dataset/foursquare/nyc/user_edges.csv',help='User Edges')
    parser.add_argument('--social-path',type=str,default='dataset/foursquare/nyc/social_relations_nyc.csv',help='Social Path')
    parser.add_argument('--time-feature', type=str, default='norm_in_day_time', help='The name of time feature in the data')
    parser.add_argument('--short-traj-thres', type=int, default=2, help='Remove over-short trajectory')
    parser.add_argument('--time-units', type=int, default=48, help='Time unit is 0.5 hour, 24/0.5=48')

    # 训练超参数
    parser.add_argument('--batch',type=int,default=16,help='Batch size.')
    parser.add_argument('--epochs', type=int,default=200,help='Number of epochs to train.')
    parser.add_argument('--lr',type=float,default=0.001,help='Initial learning rate.')
    parser.add_argument('--weight_decay',type=float,default=5e-4,help='Weight decay (L2 loss on parameters).')
    parser.add_argument('--lr-scheduler-factor',type=float,default=0.1,help='Learning rate scheduler factor')
    parser.add_argument('--early_stop_patience', type=int, default=20, help='Number of epochs to wait before early stopping if no improvement.')
    parser.add_argument('--contrast_weight', type=float, default=0.5)
    parser.add_argument('--adv_weight', type=float, default=0.2)
    parser.add_argument('--temperature', type=float, default=0.1)
    parser.add_argument('--adv_eps', type=float, default=0.3, help='对抗扰动强度')

    # 模型超参数
    parser.add_argument('--gcn-nhid', type=list, default=[32, 64], help='List of hidden dims for gcn layers')
    parser.add_argument('--gcn-dropout', type=float,default=0.3, help='Dropout rate for gcn')
    parser.add_argument('--time-embed-dim', type=int, default=32, help='Time embedding dimensions')
    parser.add_argument('--dist-embed-dim', type=int, default=32, help='Distance embedding dimensions')
    parser.add_argument('--poi-embed-dim', type=int,default=128,help='POI embedding dimensions')
    parser.add_argument('--transformer-nhid', type=int, default=1024, help='Hid dim in TransformerEncoder')
    parser.add_argument('--transformer-nlayers', type=int, default=2, help='Num of TransformerEncoderLayer')
    parser.add_argument('--transformer-nhead', type=int, default=2, help='Num of heads in multiheadattention')
    parser.add_argument('--transformer-dropout', type=float, default=0.3, help='Dropout rate for transformer')
    parser.add_argument('--max-seq-length', type=int, default=50, help='Max Sequence length')
    
    return parser.parse_args()
