import os
from PIL import Image
from torch.utils.data import Dataset
import cv2
import torch
import random

class VideoDataset(Dataset):
    def __init__(self, root_dir, detector_name, transform=None, split_file=None, clip_size=8, clips_sampled=2):
        self.root_dir = root_dir
        self.transform = transform
        self.clip_size = clip_size
        self.clips_sampled = clips_sampled
        self.detector_name = detector_name
        self.video_clip_index = []  # List of (video_folder, start_idx)
        self.video_frames = {}  # Cache frame file lists
        for folder_name in split_file:
            full_path = os.path.join(self.root_dir, folder_name)
            if not os.path.isdir(full_path):
                continue

            frame_files = sorted([
                os.path.join(full_path, f) for f in os.listdir(full_path)
                if os.path.isfile(os.path.join(full_path, f))
            ])
            self.video_frames[folder_name] = frame_files

            total_frames = len(frame_files)
            if total_frames < self.clip_size:
                self.video_clip_index.append((folder_name, 0))  # One padded clip
            else:
                max_start = total_frames - self.clip_size
                possible_starts = list(range(max_start + 1))
          

                if len(possible_starts) <= self.clips_sampled:
                    selected_starts = possible_starts
                   
                else:
                    selected_starts = sorted(random.sample(possible_starts, self.clips_sampled))

                for start in selected_starts:
                    self.video_clip_index.append((folder_name, start))

    def __len__(self):
        return len(self.video_clip_index)

    def __getitem__(self, idx):
        video_folder, start_idx = self.video_clip_index[idx]
        frame_files = self.video_frames[video_folder]
     
        # Handle short videos
        if start_idx + self.clip_size > len(frame_files):
            clip_files = frame_files + [frame_files[-1]] * (self.clip_size - len(frame_files))
        else:
            clip_files = frame_files[start_idx:start_idx + self.clip_size]
            
        frames = []
        for frame_path in clip_files:
            image = cv2.imread(frame_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            if self.transform:
                image = self.transform(Image.fromarray(image).convert('RGB'))
            frames.append(image)

        clip = torch.stack(frames, dim=0)  # [T, C, H, W]
        if self.detector_name =="TallSwin":
            clip = clip.permute(1, 0, 2, 3).contiguous()  # [C, T, H, W]
            clip = clip.view(-1, clip.size(2), clip.size(3))  # [C*T, H, W]
        label = 0 if video_folder.startswith("Real") else 1
        return {
            'img': clip,
            'label': label,
            'video_id': video_folder,
            'clip_index': start_idx
        }

