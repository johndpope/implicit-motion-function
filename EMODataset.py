from moviepy.editor import VideoFileClip, ImageSequenceClip
from PIL import Image
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset
import json
import os
from typing import List, Tuple, Dict, Any
from decord import VideoReader, cpu
from rembg import remove
import io
import numpy as np
import decord
import subprocess
from tqdm import tqdm
import cv2
from pathlib import Path
from torchvision.transforms.functional import to_pil_image, to_tensor
import random
import torch.nn.functional as F
# face warp
from skimage.transform import PiecewiseAffineTransform, warp
import face_recognition
# 3dmm
import mediapipe as mp
import cv2
def gpu_padded_collate(batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
    assert isinstance(batch, list), "Batch should be a list"
    
    # Determine the maximum number of frames across all videos in the batch
    max_frames = max(len(item['frames']) for item in batch)
    
    # Determine the maximum dimensions across all frames in the batch
    max_height = max(max(frame.shape[1] for frame in item['frames']) for item in batch)
    max_width = max(max(frame.shape[2] for frame in item['frames']) for item in batch)
    
    # Pad and stack frames for each item in the batch
    padded_frames = []
    for item in batch:
        frames = item['frames']
        # Pad each frame to max_height and max_width
        padded_item_frames = [F.pad(frame, (0, max_width - frame.shape[2], 0, max_height - frame.shape[1])) for frame in frames]
        # Pad the number of frames to max_frames
        while len(padded_item_frames) < max_frames:
            padded_item_frames.append(torch.zeros_like(padded_item_frames[0]))
        # Stack frames for this item
        padded_frames.append(torch.stack(padded_item_frames))
    
    # Stack all padded frame tensors in the batch
    frames_tensor = torch.stack(padded_frames)
    
    # Assert the correct shape of the output tensor
    assert frames_tensor.ndim == 5, "Frames tensor should be 5D (batch, frames, channels, height, width)"
    
    return {'frames': frames_tensor}


class EMODataset(Dataset):
    def __init__(self, use_gpu: False, sample_rate: int,  width: int, height: int, img_scale: Tuple[float, float], img_ratio: Tuple[float, float] = (0.9, 1.0), video_dir: str = ".", drop_ratio: float = 0.1, json_file: str = "", stage: str = 'stage1', transform: transforms.Compose = None, remove_background=False, use_greenscreen=False, apply_crop_warping=False):
        self.sample_rate = sample_rate
        self.width = width
        self.height = height
        self.img_scale = img_scale
        self.img_ratio = img_ratio
        self.video_dir = video_dir
        self.transform = transform
        self.stage = stage
        self.pixel_transform = transform
        self.drop_ratio = drop_ratio
        self.remove_background = remove_background
        self.use_greenscreen = use_greenscreen
        self.apply_crop_warping = apply_crop_warping
        with open(json_file, 'r') as f:
            self.celebvhq_info = json.load(f)

        self.use_gpu = use_gpu

        decord.bridge.set_bridge('torch')  # Optional: This line sets decord to directly output PyTorch tensors.
        self.ctx = cpu()

        self.video_ids = list(self.celebvhq_info['clips'].keys())



         # Initialize MediaPipe Face Mesh
        # self.mp_face_detection = mp.solutions.face_detection
        # self.mp_face_mesh = mp.solutions.face_mesh
        # # Initialize FaceDetection once here
        # self.face_detection = self.mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)
        # self.face_mesh = self.mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, min_detection_confidence=0.5)

        # self.face_detection = self.mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)
        # self.face_mesh = self.mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, min_detection_confidence=0.5)

        # self.HEAD_POSE_LANDMARKS = [33, 263, 1, 61, 291, 199]
    # def __del__(self):
        # self.face_detection.close()
        # self.face_mesh.close()


    def __len__(self) -> int:
        return len(self.video_ids)

    def warp_and_crop_face(self, image_tensor, video_name, frame_idx, transform=None, output_dir="output_images", warp_strength=0.01, apply_warp=False):
        # Ensure the output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        # Construct the file path
        output_path = os.path.join(output_dir, f"{video_name}_frame_{frame_idx}.png")
        
        # Check if the file already exists
        if os.path.exists(output_path):
            # Load and return the existing image as a tensor
            existing_image = Image.open(output_path).convert("RGBA")
            return to_tensor(existing_image)
        
        # Check if the input tensor has a batch dimension and handle it
        if image_tensor.ndim == 4:
            # Assuming batch size is the first dimension, process one image at a time
            image_tensor = image_tensor.squeeze(0)
        
        # Convert the single image tensor to a PIL Image
        image = to_pil_image(image_tensor)
        
        # Remove the background from the image
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format='PNG')
        img_byte_arr = img_byte_arr.getvalue()
        bg_removed_bytes = remove(img_byte_arr)
        bg_removed_image = Image.open(io.BytesIO(bg_removed_bytes)).convert("RGBA")
        
        # Convert the image to RGB format to make it compatible with face_recognition
        bg_removed_image_rgb = bg_removed_image.convert("RGB")
        
        # Detect the face in the background-removed RGB image using the numpy array
        face_locations = face_recognition.face_locations(np.array(bg_removed_image_rgb))
        
        if len(face_locations) > 0:
            top, right, bottom, left = face_locations[0]
            
            # Automatically choose sweet spot to crop.
            # https://github.com/tencent-ailab/V-Express/blob/main/assets/crop_example.jpeg
            
            face_width = right - left
            face_height = bottom - top

            # Calculate the padding amount based on face size and output dimensions
            pad_width = int(face_width * 0.5)
            pad_height = int(face_height * 0.5)

            # Expand the cropping coordinates with the calculated padding
            left_with_pad = max(0, left - pad_width)
            top_with_pad = max(0, top - pad_height)
            right_with_pad = min(bg_removed_image.width, right + pad_width)
            bottom_with_pad = min(bg_removed_image.height, bottom + pad_height)

            # Crop the face region from the image with padding
            face_image_with_pad = bg_removed_image.crop((left_with_pad, top_with_pad, right_with_pad, bottom_with_pad))

            # Crop the face region from the image without padding
            face_image_no_pad = bg_removed_image.crop((left, top, right, bottom))
            
            if apply_warp:
                # Convert the face image to a numpy array
                face_array_with_pad = np.array(face_image_with_pad)
                face_array_no_pad = np.array(face_image_no_pad)
                
                # Generate random control points for thin-plate-spline warping
                rows_with_pad, cols_with_pad = face_array_with_pad.shape[:2]
                rows_no_pad, cols_no_pad = face_array_no_pad.shape[:2]
                src_points_with_pad = np.array([[0, 0], [cols_with_pad-1, 0], [0, rows_with_pad-1], [cols_with_pad-1, rows_with_pad-1]])
                src_points_no_pad = np.array([[0, 0], [cols_no_pad-1, 0], [0, rows_no_pad-1], [cols_no_pad-1, rows_no_pad-1]])
                dst_points_with_pad = src_points_with_pad + np.random.randn(4, 2) * (rows_with_pad * warp_strength)
                dst_points_no_pad = src_points_no_pad + np.random.randn(4, 2) * (rows_no_pad * warp_strength)
                
                # Create a PiecewiseAffineTransform object
                tps_with_pad = PiecewiseAffineTransform()
                tps_with_pad.estimate(src_points_with_pad, dst_points_with_pad)
                tps_no_pad = PiecewiseAffineTransform()
                tps_no_pad.estimate(src_points_no_pad, dst_points_no_pad)
                
                # Apply the thin-plate-spline warping to the face images
                warped_face_array_with_pad = warp(face_array_with_pad, tps_with_pad, output_shape=(rows_with_pad, cols_with_pad))
                warped_face_array_no_pad = warp(face_array_no_pad, tps_no_pad, output_shape=(rows_no_pad, cols_no_pad))
                
                # Convert the warped face arrays back to PIL images
                warped_face_image_with_pad = Image.fromarray((warped_face_array_with_pad * 255).astype(np.uint8))
                warped_face_image_no_pad = Image.fromarray((warped_face_array_no_pad * 255).astype(np.uint8))
            else:
                warped_face_image_with_pad = face_image_with_pad
                warped_face_image_no_pad = face_image_no_pad
            
            # Apply the transform if provided
            if transform:
                warped_face_image_with_pad = warped_face_image_with_pad.convert("RGB")
                warped_face_image_no_pad = warped_face_image_no_pad.convert("RGB")
                warped_face_tensor_with_pad = transform(warped_face_image_with_pad)
                warped_face_tensor_no_pad = transform(warped_face_image_no_pad)
                return warped_face_tensor_with_pad, warped_face_tensor_no_pad
            
            # Convert the warped PIL images back to tensors
            warped_face_image_with_pad = warped_face_image_with_pad.convert("RGB")
            warped_face_image_no_pad = warped_face_image_no_pad.convert("RGB")
            return to_tensor(warped_face_image_with_pad), to_tensor(warped_face_image_no_pad)

        else:
            return None, None

    def extract_face_features(self, image):
        # Convert the image to RGB (MediaPipe requires RGB input)
        image_rgb = cv2.cvtColor(np.array(image), cv2.COLOR_BGR2RGB)
        
        # Process the image
        results = self.face_mesh.process(image_rgb)
        print("results:",results)
        img_h, img_w, _ = image.shape
        face_3d = []
        face_2d = []


        if results.multi_face_landmarks:       
            for face_landmarks in results.multi_face_landmarks:
                key_landmark_positions=[]
                for idx, lm in enumerate(face_landmarks.landmark):
                    if idx in self.HEAD_POSE_LANDMARKS:
                        x, y = int(lm.x * img_w), int(lm.y * img_h)
                        face_2d.append([x, y])
                        face_3d.append([x, y, lm.z])

                        landmark_position = [x,y]
                        key_landmark_positions.append(landmark_position)
                # Convert to numpy arrays
                face_2d = np.array(face_2d, dtype=np.float64)
                face_3d = np.array(face_3d, dtype=np.float64)

                # Camera matrix
                focal_length = img_w  # Assuming fx = fy
                cam_matrix = np.array(
                    [[focal_length, 0, img_w / 2],
                    [0, focal_length, img_h / 2],
                    [0, 0, 1]]
                )

                # Distortion matrix
                dist_matrix = np.zeros((4, 1), dtype=np.float64)

                # Solve PnP to get rotation vector
                success, rot_vec, trans_vec = cv2.solvePnP(
                    face_3d, face_2d, cam_matrix, dist_matrix
                )
                yaw, pitch, roll = self.calculate_pose(key_landmark_positions)
                print(f'Roll: {roll:.4f}, Pitch: {pitch:.4f}, Yaw: {yaw:.4f}')
                self.draw_axis(image, yaw, pitch, roll)
                debug_image_path = image_path.replace('.jpg', '_debug.jpg')  # Modify as needed
                cv2.imwrite(debug_image_path, image)
                print(f'Debug image saved to {debug_image_path}')
                
                ok = torch.tensor([roll, pitch, yaw])
                return ok                 
        
        return torch.tensor([0,0,0])
    
    def load_and_process_video(self, video_path: str) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        video_id = Path(video_path).stem
        output_dir = Path(self.video_dir) / video_id
        output_dir.mkdir(exist_ok=True)
        
        tensor_frames = []
        face_features = []

        tensor_file_path = output_dir / f"{video_id}_tensors_and_features.npz"

        if tensor_file_path.exists():
            print(f"Loading processed tensors and face features from file: {tensor_file_path}")
            with np.load(tensor_file_path) as data:
                tensor_frames = [torch.tensor(data[f'frame_{i}']) for i in range(len(data) // 2)]
                face_features = [torch.tensor(data[f'features_{i}']) for i in range(len(data) // 2)]
        else:
            print(f"Processing video frames and extracting face features: {output_dir}")
            video_reader = VideoReader(video_path, ctx=self.ctx)
            for frame_idx in tqdm(range(len(video_reader)), desc="Processing Video Frames"):
                frame = Image.fromarray(video_reader[frame_idx].numpy())
                tensor_frame, image_frame = self.augmentation(frame, self.pixel_transform)
                
                # Extract face features
                features =   torch.tensor([0.0, 0.0, 0.0])
                # features = self.extract_face_features(image_frame)
                
                tensor_frames.append(tensor_frame)
                face_features.append(features)

                # Save frame as PNG image
                image_frame.save(output_dir / f"{frame_idx:06d}.png")

            # Save tensors and features
            save_dict = {}
            for i, (frame, feature) in enumerate(zip(tensor_frames, face_features)):
                save_dict[f'frame_{i}'] = frame.numpy()
                save_dict[f'features_{i}'] = feature.numpy()
            np.savez_compressed(tensor_file_path, **save_dict)
            print(f"Processed tensors and face features saved to file: {tensor_file_path}")

        return tensor_frames, face_features

    def augmentation(self, images, transform, state=None):
        if state is not None:
            torch.set_rng_state(state)

        if isinstance(images, list):
            if self.remove_background:
                images = [self.remove_bg(img) for img in images]
            transformed_images = [transform(img) for img in tqdm(images, desc="Augmenting Images")]
            ret_tensor = torch.stack(transformed_images, dim=0)
        else:
            if self.remove_background:
                images = self.remove_bg(images)
            ret_tensor = transform(images)

        return ret_tensor, images

    def remove_bg(self, image):
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format='PNG')
        img_byte_arr = img_byte_arr.getvalue()
        bg_removed_bytes = remove(img_byte_arr)
        bg_removed_image = Image.open(io.BytesIO(bg_removed_bytes)).convert("RGBA")  # Use RGBA to keep transparency

        if self.use_greenscreen:
            # Create a green screen background
            green_screen = Image.new("RGBA", bg_removed_image.size, (0, 255, 0, 255))  # Green color

            # Composite the image onto the green screen
            final_image = Image.alpha_composite(green_screen, bg_removed_image)
        else:
            final_image = bg_removed_image

        final_image = final_image.convert("RGB")  # Convert to RGB format
        return final_image

    def save_video(self, frames, output_path, fps=30):
        print(f"Saving video with {len(frames)} frames to {output_path}")

        # Define the codec and create VideoWriter object
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # You can change 'mp4v' to other codecs if needed
        height, width, _ = np.array(frames[0]).shape
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        for frame in frames:
            frame = np.array(frame)
            if frame.shape[2] == 4:
                frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2RGB)
            out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))  # Convert to BGR format

        out.release()
        print(f"Video saved to {output_path}")

    def process_video(self, video_path):
        processed_frames = self.process_video_frames(video_path)
        return processed_frames

    def process_video_frames(self, video_path: str) -> List[torch.Tensor]:
        video_reader = VideoReader(video_path, ctx=self.ctx)
        processed_frames = []
        for frame_idx in tqdm(range(len(video_reader)), desc="Processing Video Frames"):
            frame = Image.fromarray(video_reader[frame_idx].numpy())
            state = torch.get_rng_state()
            tensor_frame, image_frame = self.augmentation(frame, self.pixel_transform, state)
            processed_frames.append(image_frame)
        return processed_frames

    def __getitem__(self, index: int) -> Dict[str, Any]:
        video_id = self.video_ids[index]
        vid_pil_image_list, face_features_list = self.load_and_process_video(os.path.join(self.video_dir, f"{video_id}.mp4"))

        sample = {
            "video_id": video_id,
            "frames": vid_pil_image_list,
            "num_frames": len(vid_pil_image_list),
            "face_features": face_features_list
        }
        return sample