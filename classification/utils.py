import os
import numpy as np
import random
import torch
import importlib
from torch import optim
from sklearn.metrics import roc_curve
import pandas as pd
from perturbations import (apply_gaussian_blur,add_gaussian_noise, apply_color_contrast, apply_color_brightness, apply_color_saturation,
                            jpeg_compression, video_compression, rotate_image, change_resolution,add_block_distractors, add_text_distractor,
                            add_logo_distractor, add_shape_distractors,generate_random_color, drop_random_frames, change_frame_rate,
                            generate_random_text,generate_fixed_shapes,create_logo)

from functools import partial

def get_dirs(dir, dataset, category, split_type, perturbed = 0, cross_dataset = "None"): 

    
    img_folder="images"
    data_split_all = []
    #Cross-dataset
    if cross_dataset =="GenVideo":
        folders = os.listdir(os.path.join(dir,dataset,"images"))
        folders = [f"{dataset}/images/{element}" for element in folders]
        # Sample 700 only if there are more than 700
        if len(folders) > 700:
            folders = random.sample(folders, 700)
        return folders
    elif cross_dataset =="Vahdati":
        folders = os.listdir(os.path.join(dir,dataset,"images"))
        folders = [f"{dataset}/images/{element}" for element in folders]
        # Sample 400 only if there are more than 400
        if len(folders) > 400:
            folders = random.sample(folders, 400)
        return folders
    elif cross_dataset =="GenVidBench":
        folders = os.listdir(os.path.join(dir,dataset,"images"))
        folders = [f"{dataset}/images/{element}" for element in folders]
        # Sample 700 only if there are more than 700
        if len(folders) > 700:
            folders = random.sample(folders, 700)
        return folders
    #DecepTIV
    else:
        all_categories = ["Firefighter", "Weather", "Soldier"]
        all_datasets = ["HunyuanVideo","Open-Sora","EasyAnimate_I2V","EasyAnimate_T2V","DynamiCrafter", "SVD",
                        "CogVideo_I2V", "CogVideo_T2V", "Luma", "Gen3","Wan2.1","Veo3-T2V", "Veo3-I2V", "Sora2-T2V", "Sora2-I2V"]
        test_only = ["CogVideo_I2V", "CogVideo_T2V", "Luma", "Gen3","Wan2.1","Veo3-T2V", "Veo3-I2V", "Sora2-T2V", "Sora2-I2V"]
        
        if split_type in ("train", "val") and dataset=="all":
            for k in test_only:
                all_datasets.remove(k)
            all_datasets = all_datasets
    
    if perturbed==1:
        img_folder = f"images_pert"
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
    
    return None

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
    elif opt_name == 'adam':
        optimizer = optim.Adam(
            params=model.parameters(),
            lr=config['optimizer'][opt_name]['lr'],
            weight_decay=config['optimizer'][opt_name]['weight_decay'],
            betas=(config['optimizer'][opt_name]['beta1'], config['optimizer'][opt_name]['beta2']),
            eps=config['optimizer'][opt_name]['eps'],
            amsgrad=config['optimizer'][opt_name]['amsgrad'],
        )
    else:
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

def load_checkpoint(config, model, base_dir, ckpt_dir):

    model_name = config['model']['name']
    if model_name == "Clip_VPT":
        model_name = model_name  +"_" + config["model"]["type"] + str(config["backbone"]["prompt_length"])
    ckpt_path = os.path.join(
        base_dir, "classification/ckpts/",
        model_name, ckpt_dir,
        config['ckpt_test']
    )
    ckpt = torch.load(ckpt_path, map_location="cpu")
    if config['model']['name'] =="Clip_VPT":

        state_dict = ckpt["model"]
        # Load prompts
        if hasattr(model, "prompts"):
            model.prompts = torch.nn.Parameter(state_dict['prompts'])
        # Load linear layer
        model.fc.load_state_dict(state_dict['fc'], strict=True)
    elif model_name == "Universal_FD":
     
        state_dict = ckpt["model"]
        model.fc.load_state_dict(state_dict['fc'], strict=True)
    elif model_name == "UNITE":
     
        state_dict = ckpt["model"]
        # Load video transformer weights
        model.vid_transformer.load_state_dict(state_dict['video_transformer'], strict=True)
        # Load final classifier weights

        model.fc.load_state_dict(state_dict['fc'], strict=True)
    else:
        # Case 3: End-to-end trained model
        model.load_state_dict(ckpt["model"], strict=True)

    return model



def choose_scheduler(config, optimizer):
    if config['lr_scheduler'] is None:
        return None
    elif config['lr_scheduler'] == 'step':
        scheduler = optim.lr_scheduler.StepLR(
            optimizer,
            step_size=config['lr_step'],
            gamma=config['lr_gamma'],
        )
        return scheduler
    elif config['lr_scheduler'] == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config['lr_T_max'],
            eta_min=config['lr_eta_min'],
        )
        return scheduler
    elif config['lr_scheduler'] == 'linear':
        scheduler = LinearDecayLR(
            optimizer,
            config['nEpochs'],
            int(config['nEpochs']/4),
        )
    else:
        raise NotImplementedError('Scheduler {} is not implemented'.format(config['lr_scheduler']))

def EER(y_true, y_scores):
    fpr, tpr, thresholds = roc_curve(y_true, y_scores, pos_label=1)
    fnr = 1 - tpr
    abs_diffs = np.abs(fpr - fnr)
    idx = np.nanargmin(abs_diffs)
    eer = (fpr[idx] + fnr[idx]) / 2  # Average at crossover
    return eer

def get_pert_function(pert_type, pert_param):
    
    func_dict = dict()  # a dict of function

    func_dict['GB'] = apply_gaussian_blur
    func_dict['GN'] = add_gaussian_noise
    func_dict['CC'] = apply_color_contrast
    func_dict['CB'] = apply_color_brightness
    func_dict['CS'] = apply_color_saturation
    func_dict['JPEG'] = jpeg_compression
    func_dict['VC'] = video_compression
    func_dict['RV'] = rotate_image
    func_dict['CR'] = change_resolution

    func_dict['BW'] = add_block_distractors
    if pert_type=="TD":
        text_x = 0
        text_y = random.randint(30, 400)
        position = (text_x, text_y)
        text_color = generate_random_color()
        random_text = generate_random_text(pert_param)
        func_dict['TD'] = partial(add_text_distractor, text = random_text, position = position,
                                 text_color = text_color, font_scale=1, thickness=2)
    if pert_type=="LD":
        logo = create_logo(text="Deepfakes", font_scale=1, thickness=2, width = 5*pert_param, height=pert_param)
        x_pos = random.randint(0, 600 - logo.shape[1])
        y_pos = random.randint(0, 450 - logo.shape[0])
        func_dict['LD'] = partial(add_logo_distractor, logo=logo, x_pos = x_pos, y_pos = y_pos)
    if pert_type=="SD":
        shapes = generate_fixed_shapes(pert_param, (500,500,3))
        func_dict['SD'] = partial(add_shape_distractors, shapes = shapes)

    func_dict['DF'] = drop_random_frames
    func_dict['FRC'] = change_frame_rate
    return func_dict[pert_type]

def get_pert_parameter(type, level):
    param_dict = dict()  # a dict of list
    param_dict['GB'] = [7, 9, 13, 17, 21]  # larger, worse
    param_dict['GN'] = [0.001, 0.002, 0.005, 0.01, 0.05]  # larger, worse
    param_dict['CC'] = [0.85, 0.725, 0.6, 0.475, 0.35]  # smaller, worse
    param_dict['CB'] = [30, 60, 90, 120, 150]  # larger, worse
    param_dict['CS'] = [0.4, 0.3, 0.2, 0.1, 0.0]  # smaller, worse
    param_dict['JPEG'] = [2, 3, 4, 5, 6]  # larger, worse
    param_dict['VC'] = [30, 32, 35, 38, 40]  # larger, worse
    param_dict['RV'] = [5, 10, 15, 20, 25]  # larger, worse
    param_dict['CR'] = [0.8,0.65,0.40,0.35,0.20]  # larger, worse
    
    param_dict['BW'] = [16, 32, 48, 64, 80]  # larger, worse
    param_dict['TD'] = [7, 14, 21, 28, 35]  # larger, worse
    param_dict['LD'] = [40, 60, 80, 100, 120]  # larger, worse
    param_dict['SD'] = [2, 4, 6, 8, 10]  # larger, worse

    param_dict['DF'] = [0.1, 0.2, 0.3, 0.4, 0.5]
    param_dict['FRC'] = [0.9, 0.8, 0.7, 0.6, 0.5]
    # level starts from 1, list starts from 0
    print(f"Pert : {type}, level {level}, param{param_dict[type][level - 1]}")
    return param_dict[type][level - 1]

def save_results(results, save_dir, ckpt_dir, category, perturbation_type, perturbation_level, cross_dataset = "None"):

    df_results = pd.DataFrame(results)
    if perturbation_type !="None":
        folder = "fine_grained_pert"
        os.makedirs(os.path.join(save_dir,folder), exist_ok=True)
        result_file_name = f"{folder}/{ckpt_dir.split('/')[0]}_{ckpt_dir.split('/')[1]}_{perturbation_type}_{perturbation_level}_results.csv"
    else:

        if cross_dataset !="None":
            result_file_name = f"{ckpt_dir.split('/')[0]}_{ckpt_dir.split('/')[1]}_results_{cross_dataset}.csv"
        else:
            result_file_name = f"{ckpt_dir.split('/')[0]}_{ckpt_dir.split('/')[1]}_results.csv"

    result_file_path = os.path.join(save_dir, result_file_name)
    # Load existing results if available
    if os.path.exists(result_file_path):
        df_existing = pd.read_csv(result_file_path)
    else:
        df_existing = pd.DataFrame(columns=["Dataset", "Test Category", "AUC-ROC","Fake Accuracy", "Real Accuracy", "Accuracy",
                                            "F1 Score", "EER", "AP", "Precision"])
    # Update or append results
    for new_result in results:
 
        mask = df_existing["Dataset"] == new_result["Dataset"]
        if mask.any():
            df_existing.loc[mask, "Test Category"] = new_result["Test Category"]
            df_existing.loc[mask, "AUC-ROC"] = new_result["AUC-ROC"]
            df_existing.loc[mask, "Fake Accuracy"] = new_result["Fake Accuracy"]
            df_existing.loc[mask, "Real Accuracy"] = new_result["Real Accuracy"]
            df_existing.loc[mask, "Accuracy"] = new_result["Accuracy"]
            df_existing.loc[mask, "F1 Score"] = new_result["F1 Score"]
            df_existing.loc[mask, "EER"] = new_result["EER"]
            df_existing.loc[mask, "AP"] = new_result["AP"]
            df_existing.loc[mask, "Precision"] = new_result["Precision"]
        else:
            df_existing = pd.concat([df_existing, pd.DataFrame([new_result])], ignore_index=True)
    
    # update existing dataframe
    df_results = df_existing.copy()

    # remove old mean rows
    
    df_results = df_results[~df_results["Dataset"].str.contains("MEAN")]

    if cross_dataset == "GenVideo":
        # Compute MEAN1: mean of the first 9 rows
        mean1 = df_results.iloc[:9].select_dtypes(include=np.number).mean()
        mean1["Dataset"] = "MEAN1"
        mean1 = mean1.to_frame().T
        df_results = pd.concat([df_results, mean1], ignore_index=True)
    elif cross_dataset== "Vahdati":
        # Compute MEAN1: mean of the first 6 rows
        mean1 = df_results.iloc[:6].select_dtypes(include=np.number).mean()
        mean1["Dataset"] = "MEAN1"
        mean1 = mean1.to_frame().T
        df_results = pd.concat([df_results, mean1], ignore_index=True)
    elif cross_dataset== "GenVidBench":
         # Compute MEAN1: mean of the first 8 rows
        mean1 = df_results.iloc[:8].select_dtypes(include=np.number).mean()
        mean1["Dataset"] = "MEAN1"
        mean1 = mean1.to_frame().T
        df_results = pd.concat([df_results, mean1], ignore_index=True)
    else:
        # Compute MEAN1: mean of the first 6 rows
        mean1 = df_results.iloc[:6].select_dtypes(include=np.number).mean()
        mean1["Dataset"] = "MEAN1"
        mean1["Test Category"] = category
        # Compute MEAN2: mean of the next 8 rows
        mean2 = df_results.iloc[6:15].select_dtypes(include=np.number).mean()
        mean2["Dataset"] = "MEAN2"
        mean2["Test Category"] = category
        # Compute MEAN3: mean of all rows (first 15)
        mean3 = df_results.iloc[:15].select_dtypes(include=np.number).mean()
        mean3["Dataset"] = "MEAN3"
        mean3["Test Category"] = category
        # Append means to df_results
        df_results = pd.concat([df_results, pd.DataFrame([mean1, mean2, mean3])], ignore_index=True)
    
    for col in df_results.columns:
        if col not in ["Dataset", "Test Category"]:
            df_results[col] = pd.to_numeric(df_results[col], errors='coerce')
    #Save to CSV
    df_results.to_csv(result_file_path, index=False, float_format="%.3f")

