
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import os
from ultralytics import YOLO


def enhance_frame(frame):
    """Apply additional enhancements to improve frame quality"""
    # Enhance contrast
    frame = cv2.equalizeHist(frame)

    # Optional: Apply slight Gaussian blur to reduce noise
    # frame = cv2.GaussianBlur(frame, (3,3), 0)

    # Optional: Sharpen
    kernel = np.array([[-1, -1, -1],
                       [-1, 9, -1],
                       [-1, -1, -1]])
    frame = cv2.filter2D(frame, -1, kernel)

    return frame


def process_video(video_path):
    print(f"Loading video from: {video_path}")

    cap = cv2.VideoCapture(video_path)
    frames = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Convert BGR to grayscale
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray_frame = clahe.apply(frame)
        gray_frame = cv2.GaussianBlur(gray_frame, (3, 3), 0)
        # Resize to 64x64 to match in_shape
        frame = cv2.resize(gray_frame, (512, 512))  # Changed from 128x128 to 64x64
        # frame = enhance_frame(frame)

        # Normalize to [0,1]
        frame = frame / 255.0

        frames.append(frame)

    cap.release()

    frames = np.array(frames, dtype=np.float32)  # Shape: [N, H, W]
    frames = frames[:, np.newaxis, :, :]  # Shape: [N, C=1, H, W]

    print(f"Processed frames shape: {frames.shape}")
    return frames


import cv2
import numpy as np
from ultralytics import YOLO


def process_video_with_yolo(video_path, yolo_model_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    frame_buffer = []

    model = YOLO(yolo_model_path)
    model.verbose = False
    output_dir = "../visualizations"
    os.makedirs(output_dir, exist_ok=True)
    window_size = 8

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

        if len(frame_buffer) == window_size:
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
            combined_frames = np.hstack([fb[0] for fb in frame_buffer])
            combined_frames = cv2.cvtColor(combined_frames, cv2.COLOR_BGR2RGB)
            output_combined_path = os.path.join(output_dir, f'aggregated_{aggregated_count}_combined.png')
            cv2.imwrite(output_combined_path, combined_frames)

            # Save each individual frame in the window for analysis
            for i, (frame_img, _) in enumerate(frame_buffer):
                frame_img = cv2.cvtColor(frame_img, cv2.COLOR_BGR2RGB)
                output_frame_path = os.path.join(output_dir, f'aggregated_{aggregated_count}_frame_{i}.png')
                cv2.imwrite(output_frame_path, frame_img)

            # Also save the mask for debugging
            output_mask_path = os.path.join(output_dir, f'aggregated_{aggregated_count}_mask.png')
            cv2.imwrite(output_mask_path, refined_mask)

            # Save the final aggregated frame (the full color frame)
            output_aggregated_path = os.path.join(output_dir, f'aggregated_{aggregated_count}.png')
            processed_frame = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
            cv2.imwrite(output_aggregated_path, (processed_frame * 255).astype(np.uint8))

            aggregated_count += 1
            frame_buffer.clear()  # Reset buffer for the next batch

        frame_count += 1

    cap.release()

    frames = np.array(frames, dtype=np.float32)
    frames = frames.transpose(0, 3, 1, 2)  # Shape: [N, C=3, H, W]

    print("Actual frames:", frame_count)
    print("Total processed aggregated frames:", len(frames))
    print("Processed frames shape:", frames.shape)
    return frames


class TrafficVideoDataset(Dataset):
    def __init__(self, data, input_frames=20, output_frames=20):
        super(TrafficVideoDataset, self).__init__()
        self.data = torch.tensor(data, dtype=torch.float32)
        self.input_frames = input_frames
        self.output_frames = output_frames

        # Add mean and std attributes
        self.mean = torch.mean(self.data).item()
        self.std = torch.std(self.data).item()

        print(f"Dataset initialized with shape: {self.data.shape}")
        print(f"Dataset mean: {self.mean}, std: {self.std}")

    def __len__(self):
        return max(0, len(self.data) - (self.input_frames + self.output_frames))

    def __getitem__(self, index):
        # Get sequences from the dataset
        input_sequence = self.data[index:index + self.input_frames]
        target_sequence = self.data[index + self.input_frames:index + self.input_frames + self.output_frames]

        # Ensure we're not accidentally using the same frames for input and target
        assert not torch.equal(input_sequence[-1], target_sequence[0]), "Input and target sequences overlap"

        # No dimension adjustment needed now - data should already be [T, C, H, W]
        return input_sequence, target_sequence


def load_data(batch_size, val_batch_size, data_root, num_workers):
    video_path = os.path.join(data_root, 'traffic/video.mp4')
    yolo_model_path = os.path.join(data_root, 'traffic/best_1.pt')

    # Process video with YOLO-based detection
    frames = process_video_with_yolo(video_path, yolo_model_path)

    # Split data
    train_size = int(0.7 * len(frames))
    val_size = int(0.15 * len(frames))
    test_size = len(frames) - train_size - val_size

    train_frames = frames[:train_size]
    val_frames = frames[train_size:train_size + val_size]
    test_frames = frames[train_size + val_size:]

    # Create datasets
    train_dataset = TrafficVideoDataset(train_frames, input_frames=10, output_frames=10)
    val_dataset = TrafficVideoDataset(val_frames, input_frames=10, output_frames=10)
    test_dataset = TrafficVideoDataset(test_frames, input_frames=10, output_frames=10)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size,
        shuffle=True, num_workers=num_workers, pin_memory=True
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=val_batch_size,
        shuffle=False, num_workers=num_workers, pin_memory=True
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=val_batch_size,
        shuffle=False, num_workers=num_workers, pin_memory=True
    )

    return train_loader, val_loader, test_loader, 0, 1


if __name__ == "__main__":
    train_loader, val_loader, test_loader, mean, std = load_data(
        batch_size=4, val_batch_size=4, data_root='./data', num_workers=0
    )