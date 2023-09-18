import argparse

def params():
    parser = argparse.ArgumentParser(description='transformer parameters')
    parser.add_argument('--save_path', type=str, default='./checkpoints/', help='location of model checkpoints')
    parser.add_argument('--data_path', type=str, default='./data/')

    parser.add_argument('--is_train', type=bool, default=True, help='if True is train model')
    parser.add_argument('--learning_rate', type=float, default=6e-4, help='original leaning rate')
    parser.add_argument('--warmup', type=int, default=100, help='warmup steps')

    parser.add_argument('--vocab_size', type=int, default=100, help='action vocab_size')
    parser.add_argument('--block_size', type=int, default=30, help='block_size = 3 * context_length')

    parser.add_argument('--train_epochs', type=int, default=0, help='total train epoch')
    parser.add_argument('--batch_size', type=int, default=4, help='batch size')
    parser.add_argument('--drop_last', type=bool, default=True)
    parser.add_argument('--input_len', type=int, default=20, help='')
    parser.add_argument('--pred_len', type=int, default=10, help='')

    parser.add_argument('--d_model', type=int, default=128, help='dimension of model')
    parser.add_argument('--n_head', type=int, default=8, help='num of heads')
    parser.add_argument('--n_layer', type=int, default=6, help='num of encoder layers')
    parser.add_argument('--embd_pdrop', type=float, default=0.1, help='pdrop of embedding')
    parser.add_argument('--resid_pdrop', type=float, default=0.1, help='pdrop of residual')
    parser.add_argument('--attn_pdrop', type=float, default=0.1, help='pdrop of attention')

    parser.add_argument('--betas', type=tuple, default=(0.9, 0.95), help='')
    parser.add_argument('--grad_norm_clip', type=float, default=1.0, help='')
    parser.add_argument('--weight_decay', type=bool, default=False, help='only applied on matmul weights')

    parser.add_argument('--lr_decay', type=bool, default=True, help='learning rate decay params: linear warmup followed by cosine decay to 10 percent of original')
    parser.add_argument('--warmup_tokens', type=int, default=512*20, help='lr adjust warmup')
    parser.add_argument('--final_tokens', type=int, default=260e9, help='lr adjust final of 0.1 original LR')
    parser.add_argument('--max_timestep', type=int, default=50000, help='')
    parser.add_argument('--num_workers', type=int, default=0, help='better be 0 in windows system')

    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--model_type', type=str, default='reward_conditioned')

    args = parser.parse_args()

    return args

