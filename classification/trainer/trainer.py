from tqdm import tqdm
from sklearn.metrics import roc_auc_score
import numpy as np
import os
import re
import torch

class Trainer(object):
    def __init__(
        self,
        config,
        model,
        optimizer,
        dataset,
        category,
        balanced
        ):

        self.config = config
        self.model = model
        self.optimizer = optimizer
        self.dataset = dataset
        self.category = category
        self.balanced = balanced

    def setTrain(self):
        self.model.train()
        self.train = True

    def setEval(self):
        self.model.eval()
        self.train = False

    def train_step(self,data_dict):
        
        x = data_dict['img'].cuda()
        labels = data_dict['label'].cuda()
        prediction_dict = self.model(x)
        loss = self.model.get_losses(labels, prediction_dict['cls'])
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss,prediction_dict

    def model_validation(self, val_loader):

        predictions = np.array([])
        real_labels = np.array([])
        with torch.no_grad():
            for batch_idx, data_dict in enumerate(val_loader):

                x = data_dict['img'].cuda()
                labels = data_dict['label'].cuda()
                prediction_dict = self.model(x)
                y_probs = prediction_dict['prob']
                predictions = np.append(predictions,y_probs.cpu().detach().numpy())
                real_labels = np.append(real_labels,labels.cpu().detach().numpy())
        val_auc = roc_auc_score(real_labels, predictions)
        return val_auc


    def save_model(self, epoch, val_auc):

        save_dir = os.path.join(self.config['save_dir'], self.config['model']['name'], self.dataset, self.category)
        os.makedirs(save_dir, exist_ok=True)  # Ensure directory exists

        # Regular expression to extract AUC score from filename
        pattern = re.compile(r'model_epoch\d+_val([\d.]+)\.tar')

        # Scan directory for existing models and extract AUC scores
        existing_models = {}
        for filename in os.listdir(save_dir):
            match = pattern.match(filename)
            if match:
                existing_auc = float(match.group(1))
                existing_models[os.path.join(save_dir, filename)] = existing_auc
        # Check if current model has the highest AUC
        if not existing_models or val_auc > max(existing_models.values()):
            # Delete all models with a lower AUC
            for file_path, auc in existing_models.items():
                if auc < val_auc:
                    os.remove(file_path)
            # Save the new model
            save_model_path = os.path.join(save_dir, f'model_epoch{epoch}_val{val_auc:.4f}.tar')
            torch.save({'model': self.model.state_dict(), 'epoch': epoch}, save_model_path)
            print(f'Saving model from epoch {epoch} with AUC {val_auc:.4f}', flush=True)

    
    def train_epoch(self,epoch,train_loader,val_loader):
        
        losses = []
        for batch_idx, data_dict in enumerate(tqdm(train_loader)):

            self.setTrain()
            loss,predictions=self.train_step(data_dict) #SGD
            losses.append(loss.item())
            if batch_idx % self.config['log_interval'] == 0:
                print(f'Loss: {np.mean(losses):.6f}')
        #Model eval
        self.setEval()
        val_auc = self.model_validation(val_loader)
        #Model saving
        self.save_model(epoch, val_auc)

