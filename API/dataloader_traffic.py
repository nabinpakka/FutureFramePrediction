import torch
from torch.utils.data import Dataset, DataLoader, random_split
import os
import cv2
import numpy as np
from ultralytics import YOLO

def save_frames_as_video(frames, video_path):
    video_name = os.path.basename(video_path)
    output_dir = os.path.dirname(video_path)

    name, ext = os.path.splitext(video_name)
    if ext.lower() not in ['.mp4', '.avi', '.mov']:
        ext = '.mp4'  # enforce a compatible format

    output = name + "_agg" + ext
    output_path = os.path.join(output_dir, output)

    first_frame = frames[0]
    if len(first_frame.shape) == 2:
        first_frame = cv2.cvtColor(first_frame, cv2.COLOR_GRAY2BGR)
    _, height, width = first_frame.shape

    fps = 24
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    for frame in frames:
        # (C,H,W) -> (H,W,C)
        frame = frame.transpose(1,2,0)
        if frame.dtype != np.uint8:
            frame = (255 * frame).clip(0, 255).astype(np.uint8)
        if len(frame.shape) == 2:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        out.write(frame)

    out.release()
    print(f"✅ Video saved as {output_path}")

def process_video_with_yolo(video_path, yolo_model_path):

    # if the video file contains "_agg", then the video is already aggregated. So, a flag to skip the aggregation part
    is_agg = False
    if "_agg" in video_path:
        is_agg = True

    cap = cv2.VideoCapture(video_path)
    frames = []
    frame_buffer = []

    model = YOLO(yolo_model_path)
    model.verbose = False
    output_dir = "/home/nabin/research/traffic/trafficeFramePrediction/FutureFramePrediction/visualizations"
    os.makedirs(output_dir, exist_ok=True)
    window_size = 30

    frame_count = 0
    aggregated_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        orig_h, orig_w = frame.shape[:2]
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Resize frame to 512x512
        frame = cv2.resize(frame, (512, 512))

        # TODO car is represented by class 2 but it is not present here
        results = model.predict(frame, classes=[3, 5, 8], conf=0.4)
        boxes = results[0].boxes
        mask = np.zeros((orig_h, orig_w), dtype=np.uint8)

        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            x1, y1, x2, y2 = max(0, x1), max(0, y1), min(orig_w, x2), min(orig_h, y2)
            mask[y1:y2, x1:x2] = 255

        frame_buffer.append((frame, mask))

        if len(frame_buffer) == window_size and not is_agg:
            # Generate the aggregated mask from all frames in the buffer
            aggregated_mask = frame_buffer[0][1]
            for i in range(1, window_size):
                aggregated_mask = cv2.bitwise_or(aggregated_mask, frame_buffer[i][1])

            refined_mask = aggregated_mask
            for i in range(window_size):
                refined_mask = cv2.bitwise_and(refined_mask, frame_buffer[i][1])

            # Normalize the last frame to [0,1] range
            last_frame = frame_buffer[-1][0].copy().astype(np.float32) / 255.0

            # We'll use the last frame as-is - this preserves the background and vehicles
            processed_frame = last_frame

            frames.append(processed_frame)

            # Visualize all frames used in aggregation (as original BGR images)
            # combined_frames = np.hstack([fb[0] for fb in frame_buffer])
            # combined_frames = cv2.cvtColor(combined_frames, cv2.COLOR_BGR2RGB)
            # output_combined_path = os.path.join(output_dir, f'aggregated_{aggregated_count}_combined.png')
            # cv2.imwrite(output_combined_path, combined_frames)

            # Save each individual frame in the window for analysis
            # for i, (frame_img, _) in enumerate(frame_buffer):
            #     frame_img = cv2.cvtColor(frame_img, cv2.COLOR_BGR2RGB)
            #     output_frame_path = os.path.join(output_dir, f'aggregated_{aggregated_count}_frame_{i}.png')
            #     cv2.imwrite(output_frame_path, frame_img)

            # Also save the mask for debugging
            # output_mask_path = os.path.join(output_dir, f'aggregated_{aggregated_count}_mask.png')
            # cv2.imwrite(output_mask_path, refined_mask)

            # Save the final aggregated frame (the full color frame)
            # output_aggregated_path = os.path.join(output_dir, f'aggregated_{aggregated_count}.png')
            # processed_frame = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
            # cv2.imwrite(output_aggregated_path, (processed_frame * 255).astype(np.uint8))

            aggregated_count += 1
            frame_buffer.clear()  # Reset buffer for the next batch
        else:
            frames.append(frame)
        frame_count += 1

    cap.release()

    frames = np.array(frames, dtype=np.float32)
    frames = frames.transpose(0, 3, 1, 2)  # Shape: [N, C=3, H, W]

    if not is_agg:
        save_frames_as_video(frames, video_path)

    print("Actual frames:", frame_count)
    print("Total processed aggregated frames:", len(frames))
    print("Processed frames shape:", frames.shape)
    return frames

def process_video(
    yolo_model_path: str,
    video_path: str,
    window_size: int = 30,
    iou_thresh: float = 0.3,
    output_dir: str = "./visualizations"
):
    """
    For each window of `window_size` frames, produces one composite image
    showing every vehicle (crisp, full-color) overlaid on the true background.
    """
    model = YOLO(yolo_model_path)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    mask_dir = os.path.join(output_dir, "masks")
    img_dir = os.path.join(output_dir, "composites")
    inter_dir = os.path.join(output_dir, "inter_masks")
    os.makedirs(mask_dir,  exist_ok=True)
    os.makedirs(img_dir,   exist_ok=True)
    os.makedirs(inter_dir, exist_ok=True)

    def compute_iou(b1, b2):
        x1, y1, x2, y2   = b1
        x1p, y1p, x2p, y2p = b2
        xi1, yi1 = max(x1, x1p), max(y1, y1p)
        xi2, yi2 = min(x2, x2p), min(y2, y2p)
        if xi2 <= xi1 or yi2 <= yi1:
            return 0.0
        inter = (xi2 - xi1) * (yi2 - yi1)
        a1 = (x2 - x1) * (y2 - y1)
        a2 = (x2p - x1p) * (y2p - y1p)
        return inter / (a1 + a2 - inter)

    frames, detections = [], []
    batch = 0
    model_input_frames = []

    while True:

        ret, frame = cap.read()
        if not ret:
            break

        frames.append(frame)
        res = model.predict(frame, classes=[3,5,8], conf=0.4, device=1)
        boxes = [
            (int(b.xyxy[0][0]), int(b.xyxy[0][1]),
             int(b.xyxy[0][2]), int(b.xyxy[0][3]))
            for b in res[0].boxes
        ]
        detections.append(boxes)

        if len(frames) == window_size:
            batch += 1
            h, w  = frames[0].shape[:2]
            
           
            all_frames = np.array(frames)
            background = np.median(all_frames, axis=0).astype(np.uint8)
            
            full_masks = []
            prev_boxes = detections[0]
            prev_mask  = np.zeros((h, w), dtype=np.uint8)

            for i in range(1, window_size):
                curr_boxes    = detections[i]
                inter_mask    = np.zeros((h, w), dtype=np.uint8)
                curr_box_mask = np.zeros((h, w), dtype=np.uint8)

                for pb in prev_boxes:
                    for cb in curr_boxes:
                        if compute_iou(pb, cb) > iou_thresh:
                            xi1, yi1 = max(pb[0], cb[0]), max(pb[1], cb[1])
                            xi2, yi2 = min(pb[2], cb[2]), min(pb[3], cb[3])
                            prev_mask[pb[1]:pb[3], pb[0]:pb[2]] = 0
                        else:
                            prev_mask[pb[1]:pb[3], pb[0]:pb[2]] = 255

                for (x1,y1,x2,y2) in curr_boxes:
                    curr_box_mask[y1:y2, x1:x2] = 255

                step_mask = cv2.bitwise_or(prev_mask, curr_box_mask)
                full_masks.append(step_mask.copy())

                prev_boxes = curr_boxes
                prev_mask  = step_mask

            composite = np.zeros_like(frames[0])
            for i, mask in enumerate(full_masks, start=1):
                
                cv2.copyTo(frames[i], mask, composite)
         
                cv2.imwrite(
                    os.path.join(inter_dir, f"batch{batch:03d}_step{i:03d}.png"),
                    mask
                )

            gray      = cv2.cvtColor(composite, cv2.COLOR_BGR2GRAY)
            ys, xs    = np.where(gray > 0)
            final_img = background.copy()
            final_img[ys, xs] = composite[ys, xs]

            cv2.imwrite(os.path.join(img_dir,  f"comp_batch_{batch:03d}.png"),
                        final_img)
            
            # final_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            model_frame = cv2.resize(final_img, (512, 512))
            model_frame = np.clip(model_frame / 255.0, 0, 1)
            model_input_frames.append(model_frame)
          
            print(f"[Batch {batch:03d}] composite → {img_dir}, mask → {mask_dir}")

            frames.clear()
            detections.clear()

    cap.release()

    model_input_frames = np.array(model_input_frames, dtype=np.float32)
    model_input_frames = model_input_frames.transpose(0, 3, 1, 2)
    print(f"Returning {len(model_input_frames)} frames for model input with shape {model_input_frames.shape}")
    return model_input_frames



class TrafficVideoDataset(Dataset):
    def __init__(self, data, input_frames=20, output_frames=20, lead_time=10):
        super(TrafficVideoDataset, self).__init__()
        self.data = torch.tensor(data, dtype=torch.float32)
        self.input_frames = input_frames
        self.output_frames = output_frames
        self.lead_time = lead_time

        self.step = 2  # STEP SIZE
        
        # Add mean and std attributes
        self.mean = torch.mean(self.data).item()
        self.std = torch.std(self.data).item()
        
        print(f"Dataset initialized with shape: {self.data.shape}")
        print(f"Dataset mean: {self.mean}, std: {self.std}")
        
    def __len__(self):
        total_frames = self.input_frames + self.output_frames + self.lead_time
        return max(0, (len(self.data) - total_frames) // self.step)
        # return max(0, len(self.data) - (self.input_frames + self.output_frames + self.lead_time))
    
    def __getitem__(self, index):
        index = index * self.step  # Use step size to skip frames
        print("The index is: ", index)

        # Get sequences from the dataset
        input_sequence = self.data[index:index + self.input_frames]

        # adjusting the frame indexes based on the lead time
        # if (len(self.data) - 1) < index + self.input_frames + self.lead_time + self.output_frames:
        #     index = index
        target_start =  index + self.input_frames + self.lead_time
        target_end = target_start + self.output_frames

        # changes made to predict only 2 frames with overlapping
        target_sequence = self.data[target_start: target_end]

        #TODO asserting no overlapping but we want overlapping
        # assert not torch.equal(input_sequence[-1], target_sequence[0]), "Input and target sequences overlap"
        
        return input_sequence, target_sequence

def load_data(batch_size, val_batch_size, data_root, num_workers, input_frames, output_frames):
    video_path = os.path.join(data_root, 'traffic/vid_agg.mp4')

    # TODO either retrain the YOLO model for the blurr frames or make frames more readable
    yolo_model_path = os.path.join(data_root, 'traffic/best_1.pt')

    
    # Process video with YOLO-based detection
    frames = process_video_with_yolo( video_path, yolo_model_path)

    
    print(f"DEBUG: Total frames processed: {len(frames)}")
    
    lead_time = 30
    # Split data
    train_size = int(0.6 * len(frames))
    val_size = int(0.2 * len(frames))
    test_size = len(frames) - train_size - val_size
    
    train_frames = frames[:train_size - lead_time]
    val_frames = frames[train_size:train_size + val_size - lead_time]
    test_frames = frames[train_size + val_size: -lead_time]
    
    print(f"DEBUG: Split sizes - Train: {len(train_frames)}, Val: {len(val_frames)}, Test: {len(test_frames)}")
    
    # Create datasets
    train_dataset = TrafficVideoDataset(train_frames, input_frames=input_frames, output_frames=output_frames, lead_time = lead_time)
    val_dataset = TrafficVideoDataset(val_frames, input_frames=input_frames, output_frames=output_frames, lead_time = lead_time)
    test_dataset = TrafficVideoDataset(test_frames, input_frames=input_frames, output_frames=output_frames, lead_time = lead_time)

    print(f"DEBUG: Dataset lengths - Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size,
        shuffle=False, num_workers=num_workers, pin_memory=True
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=val_batch_size,
        shuffle=False, num_workers=num_workers, pin_memory=True
    )
    
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=val_batch_size,
        shuffle=False, num_workers=num_workers, pin_memory=True
    )
    
    print(f"DEBUG: DataLoader batch counts - Train: {len(train_loader)}, Val: {len(val_loader)}, Test: {len(test_loader)}")
    
    return train_loader, val_loader, test_loader, 0, 1

# if __name__ == "__main__":
#     train_loader, val_loader, test_loader, mean, std = load_data(
#         batch_size=4, val_batch_size=4, data_root='./data', num_workers=0
#     )