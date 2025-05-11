import os
import numpy as np
import random
import torch
import importlib
from torch import optim
from sklearn.metrics import roc_curve

def get_dirs(dir, dataset, category, split_type, perturbed = 0,trainset_no=6):
    all_categories = ["Firefighter", "Weather", "Soldier"]
    all_datasets = ["HunyuanVideo", "Open-Sora", "EasyAnimate_I2V", "EasyAnimate_T2V",
                   "DynamiCrafter", "SVD","CogVideo_T2V","CogVideo_I2V", "Wan2.1", "Luma", "Gen3"]
    
    test_only = ["CogVideo_I2V", "CogVideo_T2V", "Luma", "Gen3","Wan2.1"]
    
    if split_type in ("train", "val") and dataset=="all":
        for k in test_only:
            all_datasets.remove(k)
        all_datasets = all_datasets[:trainset_no]
    if perturbed==1:
        img_folder = "images_pert"
    else:
        img_folder = "images"
    if (dataset in all_datasets) or (dataset=="Real") :

        if category in all_categories:
            
            split_dir = os.path.join(dir, dataset, category, "splits", split_type + ".txt")
            if not os.path.exists(split_dir):
                
                data_split = os.listdir(os.path.join(dir,dataset,category,img_folder))
                data_split = [f"{dataset}/{category}/{img_folder}/{element}" for element in data_split]
                return data_split  
            else:

                data_split = np.loadtxt(split_dir, dtype=str).tolist()
                data_split = [f"{dataset}/{category}/{img_folder}/{element}" for element in data_split]
                return data_split
        elif category == "all":  

            data_split_all = []
            for j in all_categories:

                split_dir = os.path.join(dir, dataset ,j, "splits", split_type + ".txt")
                if not os.path.exists(split_dir):
                    
                    data_split = os.listdir(os.path.join(dir,dataset,j,img_folder))
                    data_split = [f"{dataset}/{j}/{img_folder}/{element}" for element in data_split]
                    data_split_all.extend(data_split)
                else:  

                    data_split = np.loadtxt(split_dir, dtype=str).tolist()
                    data_split = [f"{dataset}/{j}/{img_folder}/{element}" for element in data_split]
                    data_split_all.extend(data_split)
            return data_split_all

    elif dataset =="all":

        if category in all_categories:

            data_split_all = []
            for i in all_datasets:

                split_dir = os.path.join(dir, i, category, "splits", split_type + ".txt")
                if not os.path.exists(split_dir):
                    
                    data_split = os.listdir(os.path.join(dir,i,category,img_folder))
                    data_split = [f"{i}/{category}/{img_folder}/{element}" for element in data_split]
                    data_split_all.extend(data_split)
                else:  

                    data_split = np.loadtxt(split_dir, dtype=str).tolist()
                    data_split = [f"{i}/{category}/{img_folder}/{element}" for element in data_split]
                    data_split_all.extend(data_split)
            return data_split_all

        elif category == "all":

            data_split_all = []
            for i in all_datasets:
                for j in all_categories:
                    split_dir = os.path.join(dir, i, j, "splits", split_type + ".txt")
                    if not os.path.exists(split_dir):

                        data_split = os.listdir(os.path.join(dir,i,j,img_folder))
                        data_split = [f"{i}/{j}/{img_folder}/{element}" for element in data_split]
                        data_split_all.extend(data_split)
                    else:  
                        
                        data_split = np.loadtxt(split_dir, dtype=str).tolist()
                        data_split = [f"{i}/{j}/{img_folder}/{element}" for element in data_split]
                        data_split_all.extend(data_split)
            return data_split_all
    return None  # Return None if category is invalid or dataset not found

def load_model_from_config(config):

    # Get the model name from the config
    model_name = config['model']['name']
    # Dynamically import the model class (assuming models are in a 'models' folder)
    model_module = importlib.import_module(f".{model_name.lower()}_detector", package="detectors")
    # Get the class from the module
    model_class = getattr(model_module, f"{model_name}_Detector")
    # Instantiate the model 
    model = model_class(config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    return model

def choose_optimizer(model, config):
    opt_name = config['optimizer']['type']
    if opt_name == 'sgd':
        optimizer = optim.SGD(
            params=model.parameters(),
            lr=config['optimizer'][opt_name]['lr'],
            momentum=config['optimizer'][opt_name]['momentum'],
            weight_decay=config['optimizer'][opt_name]['weight_decay']
        )
        return optimizer
    elif opt_name == 'adam':
        optimizer = optim.Adam(
            params=model.parameters(),
            lr=config['optimizer'][opt_name]['lr'],
            weight_decay=config['optimizer'][opt_name]['weight_decay'],
            betas=(config['optimizer'][opt_name]['beta1'], config['optimizer'][opt_name]['beta2']),
            eps=config['optimizer'][opt_name]['eps'],
            amsgrad=config['optimizer'][opt_name]['amsgrad'],
        )
        return optimizer
        raise NotImplementedError('Optimizer {} is not implemented'.format(config['optimizer']))
    return optimizer

def init_seed(config):
    if config['manualSeed'] is None:
        config['manualSeed'] = random.randint(1, 10000)
    random.seed(config['manualSeed'])
    np.random.seed(config['manualSeed'])
    if config['cuda']:
        torch.manual_seed(config['manualSeed'])
        torch.cuda.manual_seed_all(config['manualSeed'])

def load_checkpoint(config,model,base_dir,ckpt_dir):

    ckpt_dir  = os.path.join(base_dir,"classification/ckpts/",config['model']['name'],ckpt_dir,config['ckpt_test'])
    model.load_state_dict(torch.load(ckpt_dir)["model"])
    return model


def EER(y_true, y_scores):
    fpr, tpr, thresholds = roc_curve(y_true, y_scores, pos_label=1)
    fnr = 1 - tpr
    abs_diffs = np.abs(fpr - fnr)
    idx = np.nanargmin(abs_diffs)
    eer = (fpr[idx] + fnr[idx]) / 2  # Average at crossover
    return eer

