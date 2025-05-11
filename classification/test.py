import argparse
import os
import yaml
import torch
from sklearn.metrics import (roc_auc_score, average_precision_score, accuracy_score,
                             f1_score, precision_score)
import torch.nn.functional as F
from collections import defaultdict
from tqdm import tqdm
import numpy as np
from prepare_data import get_dataloaders_testing
from utils import load_model_from_config, init_seed, load_checkpoint, EER
import pandas as pd

def get_predictions(args,model,loader):

    model.eval()
    predictions = []
    true_labels = []
    video_ids = []
    with torch.no_grad():
            for batch_idx, data in enumerate(tqdm(loader)):  

                x = data['img'].cuda()
                labels = data['label'].cuda()
                y = model(x)
                y_probs = y['prob']
                predictions.extend(y_probs.cpu().detach().numpy())
                true_labels.extend(labels.cpu().numpy())  # True labels
                video_ids.extend(data['video_id']) 
    # Group predictions and labels by video ID
    video_predictions = defaultdict(list)
    video_true_labels = defaultdict(list)
    for vid, pred, label in zip(video_ids, predictions, true_labels):
        video_predictions[vid].append(pred)
        video_true_labels[vid].append(label)
    
    # Aggregate predictions and true labels for each video
    video_predictions = {vid: np.mean(preds) for vid, preds in video_predictions.items()}
    video_true_labels = {vid: np.mean(labels) for vid, labels in video_true_labels.items()}  # Average labels
    # Convert aggregated values to arrays
    video_preds = np.array(list(video_predictions.values()))
    video_labels = np.array(list(video_true_labels.values()))
    return video_preds, video_labels


def calculate_metrics(true_labels,predictions,dataset,category):

    pred_labels = (predictions > 0.5).astype(int)
    auc_roc = roc_auc_score(true_labels, predictions) # auc-roc
    ap = average_precision_score(true_labels, predictions)# average precision
    acc = accuracy_score(true_labels, pred_labels) # accuracy
    f1 = f1_score(true_labels, pred_labels) # f1 score
    precision = precision_score(true_labels, pred_labels) # precision
    eer = EER(true_labels, predictions)
    # Compute class-wise accuracy
    fake_mask = (true_labels == 1)
    real_mask = (true_labels == 0)
    fake_acc = accuracy_score(true_labels[fake_mask], pred_labels[fake_mask]) if fake_mask.any() else None # fake video accuracy
    real_acc = accuracy_score(true_labels[real_mask], pred_labels[real_mask]) if real_mask.any() else None # real video accuracy
    print(f"{dataset} - AUC-ROC: {auc_roc:.3f}, Fake Acc: {fake_acc:.3f}, Real Acc: {real_acc:.3f}, Acc: {acc:.3f}, F1 Score: {f1:.3f}, EER: {eer:.3f}, AP {ap:.3f}, Precision: {precision:.3f}")
    return {
        "Dataset": dataset,
        "Test Category": category,
        "AUC-ROC": auc_roc,
        "Fake Accuracy": fake_acc,
        "Real Accuracy": real_acc,
        "Accuracy": acc,
        "F1 Score": f1,
        "EER": eer,
        "AP": ap,
        "Precision": precision 
    }

def main(args):
    
    # parse options and load config
    with open(args.detector_config, 'r') as f:
        config = yaml.safe_load(f)
    config = config.copy()
    config['ckpt_test'] = args.ckpt_weights
    # init seed
    init_seed(config)
    # prepare the model (detector)
    model = load_model_from_config(config)
    model = load_checkpoint(config = config, model = model, base_dir = args.base_dir, ckpt_dir = args.ckpt_dir)
    all_datasets = ["HunyuanVideo", "Open-Sora", "EasyAnimate_I2V", "EasyAnimate_T2V",
                   "DynamiCrafter", "SVD","CogVideo_T2V","CogVideo_I2V", "Wan2.1", "Luma", "Gen3"]
    results = []
    if args.dataset == "all":
        datasets = all_datasets
    else:
        datasets = [args.dataset]

    #Make Predictions for real videos
    test_loader_real = get_dataloaders_testing(base_dir=os.path.join(args.base_dir,"Dataset"), dataset = "Real", category = args.category,
                                                frames_sampled = args.frames_sampled, perturbed = args.perturbed, config = config)
    predictions_real,true_labels_real = get_predictions(args, model, test_loader_real)

    if args.perturbed == 0:
        save_dir = os.path.join(args.base_dir, "classification", "results", config['model']['name'], "normal")
    else:
        save_dir = os.path.join(args.base_dir, "classification", "results", config['model']['name'], "pert")
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(os.path.join(save_dir,"predictions"), exist_ok=True)
    # Make Predictions for fake videos
    for dataset in datasets:
        # Get data loaders
  
        test_loader_fake = get_dataloaders_testing(base_dir= os.path.join(args.base_dir,"Dataset"), dataset = dataset, category = args.category,
                                                frames_sampled = args.frames_sampled, perturbed = args.perturbed, config = config)
        predictions_fake,true_labels_fake = get_predictions(args, model, test_loader_fake)
        result = calculate_metrics(np.concatenate([true_labels_real,true_labels_fake]),
                                    np.concatenate([predictions_real,predictions_fake]),
                                    dataset,args.category)
        df_predictions = pd.DataFrame({
        "True Label": np.concatenate([true_labels_real, true_labels_fake]),
        "Predicted Probability": np.concatenate([predictions_real, predictions_fake])
        })
        predictions_output_path = os.path.join(save_dir,"predictions",f"{args.ckpt_dir.split('/')[0]}_{args.ckpt_dir.split('/')[1]}_{dataset}.csv")
        df_predictions.to_csv(predictions_output_path, index=False)    
        results.append(result)
    # Save DataFrame to CSV
    result_file_name = f"{args.ckpt_dir.split('/')[0]}_{args.ckpt_dir.split('/')[1]}_results.csv"
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
    # Save updated results
    df_existing.to_csv(result_file_path, index=False)

if __name__ == '__main__':

    p = argparse.ArgumentParser(description="Testing of Classifier.")
    p.add_argument("--base_dir", type=str, default="/sotossta/DecepTIV", help="The base directory")
    p.add_argument('--dataset', type=str,choices= ["CogVideo_T2V", "CogVideo_I2V","Open-Sora","SVD","HunyuanVideo",
                                                   "EasyAnimate_T2V","EasyAnimate_I2V","DynamiCrafter", "Luma",
                                                   "Gen3","Wan2.1","all"])
    p.add_argument("--category", type=str, choices=["Firefighter", "Weather", "Soldier", "all"], default="Firefighter",
                   help="Category of dataset")
    p.add_argument('--perturbed', type=int,choices= [0,1],default=0)  
    p.add_argument("--frames_sampled",type=int,default=1) 
    p.add_argument('--detector_config', type=str,
                    default='/sotossta/DecepTIV/classification/configs/detectors/efficientnetb4.yaml',
                    help='path to detector YAML file')
    p.add_argument('--ckpt_dir', type=str,required=True,help='The directory to model weights')
    p.add_argument('--ckpt_weights', type=str,required=True,help='The name of model weights file')
    args = p.parse_args()
    print(args)
    main(args)