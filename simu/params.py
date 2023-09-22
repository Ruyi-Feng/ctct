import argparse

def params():
    ###### simu params ######
    parser = argparse.ArgumentParser(description='transformer parameters')

    # simulation settings
    parser.add_argument('--save_dir', type=str, default='G:\\code\\CTCT\\ctct\\results\\')
    parser.add_argument('--episode_num', type=int, default=530)
    parser.add_argument('--xlsm_path', type=str, default=r'G:\\code\\CTCT\\ctct\\results\\RM_near_train.xlsm')

    # constant settings
    parser.add_argument('--VBA_name_1', type=str, default='Sheet2.CommandButton1_Click')
    parser.add_argument('--VBA_name_2', type=str, default='Sheet2.NKthStep')

    # task settings
    parser.add_argument('--model_type', type=str, default='reward_conditioned')
    parser.add_argument('--save_path', type=str, default='./checkpoints/', help='location of model checkpoints')
    parser.add_argument('--vocab_size', type=int, default=8, help='action vocab_size')
    parser.add_argument('--block_size', type=int, default=30, help='block_size = context_length')

    # net settings
    parser.add_argument('--d_model', type=int, default=128, help='dimension of model')
    parser.add_argument('--n_head', type=int, default=8, help='num of heads')
    parser.add_argument('--n_layer', type=int, default=6, help='num of encoder layers')
    parser.add_argument('--embd_pdrop', type=float, default=0.1, help='pdrop of embedding')
    parser.add_argument('--resid_pdrop', type=float, default=0.1, help='pdrop of residual')
    parser.add_argument('--attn_pdrop', type=float, default=0.1, help='pdrop of attention')

    parser.add_argument('--top_k', type=int, default=1, help='top k select action')
    parser.add_argument('--betas', type=tuple, default=(0.9, 0.95), help='')

    args = parser.parse_args()

    return args

