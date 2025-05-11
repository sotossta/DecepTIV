import os
from PIL import Image
from torch.utils.data import Dataset
import cv2
import random
import numpy as np

class CustomDataset(Dataset):

    def __init__(self, root_dir, transform = None, split_file= None,frames_sampled=48):

        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []  # List to store paths of all images
        self.labels = []  # List to store corresponding folder names 
        folders=split_file 
        for folder_name in folders:
            folder_path = os.path.join(self.root_dir, folder_name)
            images = [os.path.join(folder_name, f) for f in sorted(os.listdir(folder_path)) if os.path.isfile(os.path.join(folder_path, f))]
            if len(images) > frames_sampled:
                images = random.sample(images, frames_sampled)
            for image in images:
                self.image_paths.append(image)
    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        
        image_path = os.path.join(self.root_dir,self.image_paths[idx])
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.transform:
            image = self.transform(Image.fromarray(np.uint8(image)).convert('RGB'))

        if self.image_paths[idx].split("/")[0]=="Real":   
            return {'img' :image, 'label': 0, 'video_id': os.path.join(self.image_paths[idx].split("/")[0],self.image_paths[idx].split("/")[-2])}
        else:
            return {'img' :image, 'label': 1, 'video_id': os.path.join(self.image_paths[idx].split("/")[0],self.image_paths[idx].split("/")[-2])}
