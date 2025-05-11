from custom_dataset import CustomDataset
from video_dataset import VideoDataset
from torch.utils.data import ConcatDataset
import torch
from utils import get_dirs
from torchvision import transforms

def get_dataloaders(base_dir, dataset, category, frames_sampled, config, balanced):
    
    #------------------------Define transformations -------------------
    TRANSFORM_IMG = transforms.Compose([
        transforms.Resize((config['resolution'], config['resolution'])),
        transforms.ToTensor(),
        transforms.Normalize(mean=config['mean'], std= config['std'])
    ])
    split_file_real_train = get_dirs(dir = base_dir, dataset = "Real", category = category, split_type = "train")
    split_file_real_val = get_dirs(dir = base_dir, dataset = "Real", category = category, split_type = "val")
    split_file_fake_train = get_dirs(dir = base_dir, dataset = dataset, category = category, split_type = "train")
    split_file_fake_val = get_dirs(dir = base_dir, dataset = dataset, category = category, split_type = "val")
    frames_sampled_real = frames_sampled
    if balanced ==0:
            frames_sampled_fake = frames_sampled_real
    else:
        if dataset in ["Open-Sora", "HunyuanVideo"]:
            frames_sampled_fake = frames_sampled_real
        elif dataset in ["EasyAnimate_T2V", "EasyAnimate_I2V"]:
            frames_sampled_fake = frames_sampled_real//2
        elif dataset in ["SVD", "DynamiCrafter"]:
            frames_sampled_fake = frames_sampled_real//3  
        else:
            frames_sampled_fake = frames_sampled_real//12
    if config['model']['name'] in ["TallSwin", "FTCN"]:

        train_dataset_real = VideoDataset(root_dir = base_dir,
                               detector_name=config['model']['name'],
                               transform=TRANSFORM_IMG,
                               split_file = split_file_real_train,
                               clip_size = config['backbone']['clip_size'],
                               clips_sampled =  frames_sampled_real          
                               )
        val_dataset_real = VideoDataset(root_dir = base_dir,
                                detector_name=config['model']['name'],
                                transform=TRANSFORM_IMG,
                                split_file = split_file_real_val,
                                clip_size = config['backbone']['clip_size'],
                                clips_sampled =  frames_sampled_real   
                                )
        train_dataset_fake = VideoDataset(root_dir = base_dir,
                                detector_name=config['model']['name'],
                                transform=TRANSFORM_IMG,
                                split_file = split_file_fake_train,
                                clip_size = config['backbone']['clip_size'],
                                clips_sampled = frames_sampled_fake 
                                )
        val_dataset_fake = VideoDataset(root_dir = base_dir,
                                detector_name=config['model']['name'],
                                transform=TRANSFORM_IMG,
                                split_file = split_file_fake_val,
                                clip_size = config['backbone']['clip_size'],
                                clips_sampled = frames_sampled_fake  
                                )   
    else:
        train_dataset_real = CustomDataset(root_dir = base_dir,
                                transform=TRANSFORM_IMG,
                                split_file = split_file_real_train,
                                frames_sampled=frames_sampled_real              
                                )
        val_dataset_real = CustomDataset(root_dir = base_dir,
                                transform=TRANSFORM_IMG,
                                split_file = split_file_real_val,
                                frames_sampled=frames_sampled_real,
                                )
        train_dataset_fake = CustomDataset(root_dir = base_dir,
                                transform=TRANSFORM_IMG,
                                split_file = split_file_fake_train,
                                frames_sampled=frames_sampled_fake,
                            
                                )
        val_dataset_fake = CustomDataset(root_dir = base_dir,
                                transform=TRANSFORM_IMG,
                                split_file = split_file_fake_val,
                                frames_sampled=frames_sampled_fake,
                                )                           
    
 
    train_dataset = ConcatDataset([train_dataset_real,train_dataset_fake])
    val_dataset = ConcatDataset([val_dataset_real,val_dataset_fake])   
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               shuffle=True,
                                               batch_size= config['train_batchsize']
                                               )
    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=config['val_batchsize']
                                             )
    
    print("DATA STATISTICS")
    print("----------------------------------")
    print(f"Real Videos: 1) Training: {len(train_dataset_real)} frames from {len(split_file_real_train)} videos 2) Validation: {len(val_dataset_real)} frames from {len(split_file_real_val)} videos ")
    print("----------------------------------")
    print(f"Fake Videos: 1) Training:  {len(train_dataset_fake)} frames from {len(split_file_fake_train)} videos 2) Validation: {len(val_dataset_fake)} frames from {len(split_file_fake_val)} videos ")
    return train_loader,val_loader

def get_dataloaders_testing(base_dir, dataset, category, frames_sampled, perturbed,config):

    #------------------------Define transformations -------------------
    TRANSFORM_IMG = transforms.Compose([
        transforms.Resize((config['resolution'], config['resolution'])),
        transforms.ToTensor(),
        transforms.Normalize(mean=config['mean'], std= config['std'])
    ])

    split_file_test = get_dirs(dir = base_dir, dataset = dataset, category = category, split_type = "test", perturbed=perturbed)
    if config['model']['name'] in ["TallSwin", "FTCN"]:

        test_dataset = VideoDataset(root_dir = base_dir,
                               detector_name=config['model']['name'],
                               transform=TRANSFORM_IMG,
                               split_file = split_file_test,
                               clip_size = config['backbone']['clip_size'],
                               clips_sampled =  frames_sampled         
                               )


    else:
        test_dataset = CustomDataset(root_dir = base_dir,
                                transform=TRANSFORM_IMG,
                                split_file = split_file_test,
                                frames_sampled=frames_sampled              
                                )
    test_loader = torch.utils.data.DataLoader(test_dataset,batch_size= config['test_batchsize'])
    return test_loader

