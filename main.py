import argparse
import time
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1" 
from exp import Exp

import torch

import warnings
warnings.filterwarnings('ignore')

def create_parser():
    parser = argparse.ArgumentParser()
    # Set-up parameters
    parser.add_argument('--device', default='cuda', type=str, help='Name of device to use for tensor computations (cuda/cpu/mps)')
    parser.add_argument('--res_dir', default='./results', type=str)
    parser.add_argument('--ex_name', default='LT_30_input30_sliding', type=str)
    parser.add_argument('--use_gpu', default=True, type=bool)
    parser.add_argument('--gpu', default=1, type=int)
    parser.add_argument('--seed', default=1, type=int)

    # dataset parameters
    parser.add_argument('--batch_size', default=1, type=int, help='Batch size')
    parser.add_argument('--val_batch_size', default=1, type=int, help='Batch size')
    parser.add_argument('--data_root', default='data/')
    parser.add_argument('--dataname', default='traffic', choices=['traffic', 'traffic_dynamic'])
    parser.add_argument('--num_workers', default=8, type=int)

    # model parameters
    parser.add_argument('--in_shape', default=[30, 3, 512, 512], type=int, nargs='*')  
    parser.add_argument('--output_frames', default=2, type=int, help="Output frames")
    parser.add_argument('--hid_S', default=64, type=int)  
    parser.add_argument('--hid_T', default=256, type=int) 
    parser.add_argument('--N_S', default=4, type=int)
    parser.add_argument('--N_T', default=8, type=int)
    parser.add_argument('--groups', default=8, type=int)

    # Training parameters
    parser.add_argument('--epochs', default=50, type=int)
    parser.add_argument('--log_step', default=1, type=int)
    parser.add_argument('--lr', default=0.0001, type=float, help='Learning rate')
    parser.add_argument('--lt', default=10, type=int, help = "Lead time in seconds")
    return parser

if __name__ == '__main__':
    start = time.time()
    args = create_parser().parse_args()
    config = args.__dict__

    exp = Exp(args)
    print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>  start <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
    exp.train(args)
    train_end = time.time()
    print('>>>>>>>>>>>>>>>>>>>>>>>>>>>> testing <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
    mse = exp.test(args)
    total_time = time.time()
    print("10 frames to predict 1 with lead time of 1 secs")
    print("Training time: ", train_end - start)
    print("Total time: ", total_time - start)
    


# if __name__ == '__main__':
#     args = create_parser().parse_args()
#     config = args.__dict__
#     exp = Exp(args)
#     checkpoint_path = "/Users/ull/Documents/GRA/SimVP-master/vis_res/Debug/checkpoints/0.pth"
#     scheduler_path = "/Users/ull/Documents/GRA/SimVP-master/vis_res/Debug/checkpoints/0.pkl"

#     exp.test_from_checkpoint(checkpoint_path, scheduler_path, args)