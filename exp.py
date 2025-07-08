import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import os.path as osp
import json
import pickle

from model import SimVP
from tqdm import tqdm
from API import *
from utils import *

import cv2
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import torch.nn as nn

import time
import torch
import psutil


class Exp:
    def __init__(self, args):
        super(Exp, self).__init__()
        self.args = args
        self.config = self.args.__dict__
        self.device = self._acquire_device()

        self.output_frames = int(self.config.get("output_frames"))

        self._preparation()
        print_log(output_namespace(self.args))

        # self._get_data()
        self._select_optimizer()
        self._select_criterion()

    # def _acquire_device(self):
    #     # Check if Metal API is available for Apple Silicon (M1/M2)
    #     if torch.backends.mps.is_available():
    #         device = torch.device("mps")
    #     else:
    #         # Fallback to CPU if Metal is not available
    #         device = torch.device("cpu")

    #     print("Device is..", device)
    #     return device

    def _acquire_device(self):
        if self.args.use_gpu:
            # change this argument to change visibility of GPU to this program
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  
            print("Device is..", device)
            print_log('Use GPU: 0')
        else:
            device = torch.device('cpu')
            print_log('Use CPU')
        print("device: ", device)
        return device

    def _preparation(self):
        # Seed
        set_seed(self.args.seed)
        # Log and checkpoint
        self.path = osp.join(self.args.res_dir, self.args.ex_name)
        check_dir(self.path)

        self.checkpoints_path = osp.join(self.path, 'checkpoints')
        check_dir(self.checkpoints_path)

        sv_param = osp.join(self.path, 'model_param.json')
        with open(sv_param, 'w') as file_obj:
            json.dump(self.args.__dict__, file_obj)

        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)
        logging.basicConfig(level=logging.INFO, filename=osp.join(self.path, 'log.log'),
                            filemode='a', format='%(asctime)s - %(message)s')
        # Prepare data
        self._get_data()
        # Build the model
        self._build_model()

    def _build_model(self):
        args = self.args
        
        # Check if we need to modify the input shape for RGB
        in_shape = list(args.in_shape)
        
        # If RGB data is used, set channel dimension to 3
        if hasattr(args, 'rgb') and args.rgb:
            in_shape[1] = 3  # Set channel dimension to 3 for RGB
            print(f"Using RGB data: Modified input shape to {in_shape}")
            
        self.model = SimVP(tuple(in_shape), args.hid_S,
                           args.hid_T, args.N_S, args.N_T).to(device=self.device)

    def _get_data(self):
        config = self.args.__dict__
        in_shape = list(config.get("in_shape"))
        input_frames = in_shape[0]

        self.train_loader, self.vali_loader, self.test_loader, self.data_mean, self.data_std = load_data(input_frames = input_frames, **config)
        self.vali_loader = self.test_loader if self.vali_loader is None else self.vali_loader

    def _select_optimizer(self):
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.args.lr)
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer, max_lr=self.args.lr, steps_per_epoch=len(self.train_loader), epochs=self.args.epochs)
        return self.optimizer

    def _select_criterion(self):
        # self.criterion = torch.nn.MSELoss()
        self.criterion = nn.SmoothL1Loss()

    def _save(self, name=''):
        torch.save(self.model.state_dict(), os.path.join(
            self.checkpoints_path, name + '.pth'))
        state = self.scheduler.state_dict()
        fw = open(os.path.join(self.checkpoints_path, name + '.pkl'), 'wb')
        pickle.dump(state, fw)

 

    def train(self, args):
        config = args.__dict__
        recorder = Recorder(verbose=True)
        
        log_file = "training_log.txt"
        with open(log_file, 'w') as log:
            # Start time tracking
            start_time = time.time()
            process = psutil.Process()  # Track system memory usage

            log.write("Epoch, GPU Memory (MB), System Memory (MB), Time Elapsed (s)\n")

            for epoch in range(config['epochs']):
                train_loss = []
                self.model.train()

                train_pbar = tqdm(self.train_loader)
                for batch in train_pbar:
                    batch_x, batch_y = batch
                    print(f"Batch x shape: {batch_x.shape}, Batch y shape: {batch_y.shape}")

                    # getting length of input frames
                    input_frames = batch_x.shape[1]
                    
                    self.optimizer.zero_grad()
                    batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                    pred_y = self.model(batch_x)



                    loss = self.criterion(pred_y[:, :batch_y.shape[1] ,:,:], batch_y)
                    train_loss.append(loss.item())
                    train_pbar.set_description('train loss: {:.4f}'.format(loss.item()))

                    loss.backward()
                    self.optimizer.step()
                    self.scheduler.step()
                    torch.cuda.empty_cache()

                # Log memory usage and time after each epoch
                if torch.cuda.is_available():
                    gpu_mem_used = torch.cuda.memory_allocated(self.device) / 1024 ** 2  # in MB
                else:
                    gpu_mem_used = 0

                # Get system memory usage
                system_memory = process.memory_info().rss / 1024 ** 2  # in MB

                # Log to the file
                elapsed_time = time.time() - start_time
                log.write(f"{epoch+1}, {gpu_mem_used:.2f}, {system_memory:.2f}, {elapsed_time:.2f}\n")

            # End time tracking
            end_time = time.time()
            total_time = end_time - start_time
            print(f"Training completed in {total_time/60:.2f} minutes")

        print(f"Training log saved to {log_file}")


    def visualize_predictions(self, inputs, trues, preds, inputs_full=None, trues_full=None, save_path=None, batch_idx=0):
        """
        Visualize and save frame predictions with separate videos and individual frame images.
        """
        # Create directories for visualizations
        video_path = osp.join(save_path, 'videos')
        image_path = osp.join(save_path, 'frames')
        os.makedirs(video_path, exist_ok=True)
        os.makedirs(image_path, exist_ok=True)
        
        # Get sequences for the specified batch
        input_frames = inputs[batch_idx].cpu().numpy()  # [T, C=1, H, W]
        true_frames = trues[batch_idx].cpu().numpy()
        pred_frames = preds[batch_idx].cpu().numpy()

        def enhance_rgb_frame(frame):
            frame = np.clip(frame, 0, 1)
            frame = (frame * 255).astype(np.uint8)
            
            # Convert to proper image format based on channels
            if frame.shape[0] == 3:  # Already RGB format (CHW)
                frame = frame.transpose(1, 2, 0)
            elif frame.shape[0] == 1:  # Grayscale
                frame = np.repeat(frame, 3, axis=0)
                frame = frame.transpose(1, 2, 0)
            
            return frame

        
        def create_video(frames, filename, fps=5):
            """Create a video from a sequence of frames"""
            height, width = frames[0, 0].shape
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(osp.join(video_path, filename), 
                                fourcc, fps, (width, height))
            
            for frame in frames:
                frame_img = enhance_rgb_frame(frame)
    
                out.write(frame_img)
            
            out.release()

        def save_frame_images(frames, prefix):
            """Save individual frames as images"""
            for i, frame in enumerate(frames):
                # Convert from [0,1] to [0,255] and to uint8
                frame_img = enhance_rgb_frame(frame)
                # Save grayscale image
                cv2.imwrite(osp.join(image_path, f'{prefix}_{batch_idx:02d}_frame_{i:03d}.png'), frame_img)

        # # Save individual frame images
        save_frame_images(input_frames, 'input')
        save_frame_images(pred_frames, 'pred')
        save_frame_images(true_frames, 'true')

        # Create comparison images
        for i in range(len(true_frames)):
            # Convert frames to uint8
            print("input frames: ", input_frames.shape)
            input_img = enhance_rgb_frame(input_frames[i])
            print("Pred frames: ", pred_frames.shape)
            pred_img = enhance_rgb_frame(pred_frames[i])
            print("True frames: ", true_frames.shape)
            true_img = enhance_rgb_frame(true_frames[i])
            
            # Create side-by-side comparison
            comparison = np.hstack([input_img, pred_img, true_img])
            # Save without converting to BGR
            cv2.imwrite(osp.join(image_path, f'comparison_{batch_idx:02d}_frame_{i:03d}.png'), comparison)

        # Create side-by-side comparison video
        height, width = input_frames[0, 0].shape
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        comparison_video = cv2.VideoWriter(osp.join(video_path, f'comparison_{batch_idx}.mp4'), 
                                         fourcc, 5, (width*3, height))
        
        for i in range(len(true_frames)):
            # Convert frames to RGB format ready for display
            input_img = enhance_rgb_frame(input_frames[i])
            pred_img = enhance_rgb_frame(pred_frames[i])
            true_img = enhance_rgb_frame(true_frames[i])

            comparison_frame = np.hstack([input_img, pred_img, true_img])
            # comparison_frame_bgr = cv2.cvtColor(comparison_frame, cv2.COLOR_RGB2BGR)
            comparison_video.write(comparison_frame)
            
            # Calculate and log metrics for this frame
            mse = np.mean((pred_frames[i, 0] - true_frames[i, 0]) ** 2)
            mae = np.mean(np.abs(pred_frames[i, 0] - true_frames[i, 0]))
            ssim_val = ssim(pred_frames[i, 0], true_frames[i, 0], data_range=1.0)
            psnr_val = psnr(pred_frames[i, 0], true_frames[i, 0], data_range=1.0)
            
            print(f"Frame {i} Metrics - MSE: {mse:.4f}, MAE: {mae:.4f}, SSIM: {ssim_val:.4f}, PSNR: {psnr_val:.4f}")
        
        comparison_video.release()

        # Save individual videos
        create_video(input_frames, f'input_sequence_{batch_idx}.mp4')
        create_video(pred_frames, f'predicted_sequence_{batch_idx}.mp4')
        create_video(true_frames, f'ground_truth_sequence_{batch_idx}.mp4')
        

    def test(self, args):
        print("\nStarting test...")
        self.model.eval()
        metrics_all = []

        print(f"Number of batches in test loader: {len(self.test_loader)}")
        
        folder_path = self.path+'/results/{}/sv/'.format(args.ex_name)
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        with torch.no_grad():
            for i, (batch_x, batch_y) in enumerate(tqdm(self.test_loader)):
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                
                pred_y = self.model(batch_x)[:,:batch_y.shape[1],:,: ]
                batch_x = batch_x[:, :batch_y.shape[1],:,:]
               
                self.visualize_predictions(
                    batch_x, batch_y, pred_y,
                    None, None,
                    folder_path, batch_idx=0
                )
                 

                pred_y_np = pred_y.cpu().numpy()
                batch_y_np = batch_y.cpu().numpy()
                
                mse = np.mean((pred_y_np - batch_y_np) ** 2)
                mae = np.mean(np.abs(pred_y_np - batch_y_np))
                
                metrics_all.append({
                    'mse': mse,
                    'mae': mae
                })

        avg_metrics = {
            'mse': np.mean([m['mse'] for m in metrics_all]),
            'mae': np.mean([m['mae'] for m in metrics_all])
        }


        print("\nTest Metrics:")
        print(f"MSE: {avg_metrics['mse']:.4f}")
        print(f"MAE: {avg_metrics['mae']:.4f}")

        return avg_metrics['mse']
    

    def vali(self, vali_loader):
        if vali_loader is None:
            print("Warning: Validation loader is None!")
            return float('inf')
                
        self.model.eval()
        vali_loss = []
        
        with torch.no_grad():
            for batch_x, batch_y in tqdm(vali_loader, desc='Validating'):
                try:
                    # Move data to device
                    batch_x = batch_x.to(self.device)
                    batch_y = batch_y.to(self.device)
                    
                    # Generate predictions
                    pred_y = self.model(batch_x)[:, :batch_y.shape[1],:,:]
                    
                    # Calculate loss
                    loss = self.criterion(pred_y, batch_y)
                    
                    # Check if loss is valid
                    if not torch.isnan(loss) and not torch.isinf(loss):
                        vali_loss.append(loss.item())
                    else:
                        print(f"Warning: Invalid loss value: {loss.item()}")
                        print(f"Pred range: [{pred_y.min():.4f}, {pred_y.max():.4f}]")
                        print(f"Target range: [{batch_y.min():.4f}, {batch_y.max():.4f}]")
                        
                except Exception as e:
                    print(f"Error during validation: {e}")
                    continue
        
        # Calculate and return average validation loss
        if len(vali_loss) > 0:
            avg_loss = np.mean(vali_loss)
            print(f"Validation - Num samples: {len(vali_loss)}, Average loss: {avg_loss:.4f}")
            return avg_loss
        else:
            print("Warning: No valid losses calculated during validation!")
            return float('inf') 