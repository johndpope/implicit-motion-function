from torch.utils.data import Dataset
from decord import VideoReader, cpu
import decord
from PIL import Image
from typing import List, Tuple, Dict, Any
import random
import torch
from torch.utils.data import IterableDataset
from decord import VideoReader, cpu
import torchvision.transforms as transforms
import random
import os

class VideoDataset(Dataset):
    def __init__(self, root_dir, transform=None, frame_skip=1):
        self.root_dir = root_dir
        self.transform = transform
        self.frame_skip = frame_skip
        self.video_files = [os.path.join(subdir, file)
                            for subdir, dirs, files in os.walk(root_dir)
                            for file in files if file.endswith('.mp4')]
        decord.bridge.set_bridge('torch')  # Optional: This line sets decord to directly output PyTorch tensors.
        self.ctx = decord.cpu()

    def __len__(self):
        return 1000  # You can adjust this value based on how many samples you want per epoch

    def augmentation(self, images, transform, state=None):
        if state is not None:
            torch.set_rng_state(state)
        if isinstance(images, list):
            transformed_images = [transform(img) for img in images]
            ret_tensor = torch.stack(transformed_images, dim=0)  # (f, c, h, w)
        else:
            ret_tensor = transform(images)  # (c, h, w)
        return ret_tensor

    def __getitem__(self, idx):
        video_path = random.choice(self.video_files)
        vr = VideoReader(video_path, ctx=self.ctx)
        
        # Ensure we can get both current and reference frames
        max_frame_idx = len(vr) - self.frame_skip - 1
        if max_frame_idx < 0:
            # If the video is too short, choose another one
            return self.__getitem__(idx)
        
        current_frame_idx = random.randint(0, max_frame_idx)
        reference_frame_idx = current_frame_idx + self.frame_skip
        
        current_frame = Image.fromarray(vr[current_frame_idx].numpy())  
        reference_frame = Image.fromarray(vr[reference_frame_idx].numpy())  

        if self.transform:
            state = torch.get_rng_state()
            current_frame = self.augmentation(current_frame, self.transform, state)
            reference_frame = self.augmentation(reference_frame, self.transform, state)

        return current_frame, reference_frame
    
    def _get_video_and_frame(self, idx):
        for video_idx, video_path in enumerate(self.video_files):
            vr = VideoReader(video_path, ctx=cpu(0))
            num_frames = max(0, len(vr) - self.frame_skip)
            if idx < num_frames:
                return video_idx, idx
            idx -= num_frames
        raise IndexError("Index out of range")

class SingleVideoIterableDataset(IterableDataset):
    def __init__(self, root_dir, transform=None, frame_skip=1, shuffle=True):
        self.root_dir = root_dir
        self.frame_skip = frame_skip
        self.shuffle = shuffle
        self.video_files = [os.path.join(subdir, file)
                            for subdir, dirs, files in os.walk(root_dir)
                            for file in files if file.endswith('.mp4')]
        random.shuffle(self.video_files)

        # Use custom transform for tensors
        self.transform = transform if transform is not None else TensorTransform((256, 256))

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            iter_start = 0
            iter_end = len(self.video_files)
        else:
            per_worker = int(math.ceil(len(self.video_files) / float(worker_info.num_workers)))
            worker_id = worker_info.id
            iter_start = worker_id * per_worker
            iter_end = min(iter_start + per_worker, len(self.video_files))

        for video_idx in range(iter_start, iter_end):
            video_path = self.video_files[video_idx]
            print(f"Processing video: {video_path}")
            vr = VideoReader(video_path, ctx=cpu(0))
            frame_indices = list(range(len(vr) - self.frame_skip))
            if self.shuffle:
                random.shuffle(frame_indices)

            for frame_idx in frame_indices:
                current_frame_idx = frame_idx
                reference_frame_idx = frame_idx + self.frame_skip

                current_frame = vr[current_frame_idx].asnumpy()
                reference_frame = vr[reference_frame_idx].asnumpy()
                
                print(f"Raw frame shapes - Current: {current_frame.shape}, Reference: {reference_frame.shape}")

                current_frame = torch.from_numpy(current_frame).float().permute(2, 0, 1)
                reference_frame = torch.from_numpy(reference_frame).float().permute(2, 0, 1)

                print(f"Tensor shapes after permute - Current: {current_frame.shape}, Reference: {reference_frame.shape}")

                # Apply transforms
                current_frame = self.transform(current_frame)
                reference_frame = self.transform(reference_frame)

                print(f"Final tensor shapes - Current: {current_frame.shape}, Reference: {reference_frame.shape}")

                yield current_frame, reference_frame

    def __len__(self):
        return len(self.video_files) * 1000  # Approximation