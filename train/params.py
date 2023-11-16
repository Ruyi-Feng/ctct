import argparse

##### 训练的参数

def params():
    parser = argparse.ArgumentParser(description='transformer parameters')
    # task settings
    parser.add_argument('--model_type', type=str, default='reward_conditioned')
    parser.add_argument('--save_path', type=str, default='./checkpoints/', help='location of model checkpoints')
    parser.add_argument('--data_path', type=str, default='./data/total/')
    parser.add_argument('--vocab_size', type=int, default=8, help='action vocab_size')
    parser.add_argument('--block_size', type=int, default=10, help='block_size = context_length')
    parser.add_argument('--c_in', type=int, default=5, help='state dimension')
    parser.add_argument('--if_noise', type=bool, default=False, help='if noise')
    parser.add_argument('--noise_rate', type=float, default=0.2, help='noise rate')
    parser.add_argument('--noise_range', type=tuple, default=(-0.3, 0.3), help='noise range')

    # training settings
    parser.add_argument('--learning_rate', type=float, default=0.00001, help='original leaning rate')
    parser.add_argument('--train_epochs', type=int, default=20, help='total train epoch')
    parser.add_argument('--batch_size', type=int, default=1024, help='batch size')
    parser.add_argument('--drop_last', type=bool, default=True)
    parser.add_argument('--if_total_rtg', type=bool, default=True, help='if True, using total rtg')
    parser.add_argument('--warmup', type=int, default=200, help='warmup steps')

    # net settings
    parser.add_argument('--d_model', type=int, default=128, help='dimension of model')
    parser.add_argument('--n_head', type=int, default=8, help='num of heads')
    parser.add_argument('--n_layer', type=int, default=6, help='num of encoder layers')
    parser.add_argument('--embd_pdrop', type=float, default=0.1, help='pdrop of embedding')
    parser.add_argument('--resid_pdrop', type=float, default=0.1, help='pdrop of residual')
    parser.add_argument('--attn_pdrop', type=float, default=0.1, help='pdrop of attention')

    parser.add_argument('--top_k', type=int, default=1, help='top k select action')
    parser.add_argument('--betas', type=tuple, default=(0.9, 0.95), help='')
    parser.add_argument('--grad_norm_clip', type=float, default=1.0, help='')
    parser.add_argument('--weight_decay', type=bool, default=False, help='only applied on matmul weights')
    parser.add_argument('--lr_decay', type=bool, default=True, help='learning rate decay params: linear warmup followed by cosine decay to 10 percent of original')
    parser.add_argument('--final_tokens', type=int, default=8000, help='lr adjust final of 0.1 original LR')
    parser.add_argument('--max_timestep', type=int, default=550, help='')

    parser.add_argument('--num_workers', type=int, default=0, help='better be 0 in windows system')
    parser.add_argument('--seed', type=int, default=123)

    args = parser.parse_args()

    return args

