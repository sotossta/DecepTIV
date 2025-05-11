import argparse
import os
import yaml
from utils import load_model_from_config, choose_optimizer, init_seed
from trainer.trainer import Trainer
from prepare_data import get_dataloaders

def main(args):

    # parse options and load config
    with open(args.detector_config, 'r') as f:
        config = yaml.safe_load(f)
    # init seed
    init_seed(config)
    # prepare the model (detector)
    model = load_model_from_config(config)
    # prepare the optimizer
    optimizer = choose_optimizer(model, config)
    #-------------------- Define dataloaders -------------------
    train_loader, val_loader= get_dataloaders(base_dir=os.path.join(args.base_dir,"Dataset"), dataset=args.dataset, category=args.category,
                                             frames_sampled=args.frames_sampled_real, balanced = args.balanced, config=config)
    # prepare the metric
    # prepare the trainer
    trainer = Trainer(config, model, optimizer, dataset=args.dataset,category = args.category,balanced=args.balanced)
    # start training
    for epoch in range(config['start_epoch'], config['nEpochs'] + 1):

        print(f"Training epoch: {epoch}")
        trainer.model.epoch = epoch
        trainer.train_epoch(
                    epoch=epoch,
                    train_loader=train_loader,
                    val_loader=val_loader,
                )

if __name__ == '__main__':

    p = argparse.ArgumentParser(description="Training of Classifier.")
    p.add_argument("--base_dir", type=str, default="/sotossta/DecepTIV", help="The base directory")
    p.add_argument('--dataset', type=str,choices= ["HunyuanVideo","Open-Sora","EasyAnimate_I2V",
                                                    "EasyAnimate_T2V","DynamiCrafter", "SVD","all"])
    p.add_argument("--category", type=str, choices=["Firefighter", "Weather", "Soldier", "all"], default="Firefighter",
                   help="Category of dataset")
    p.add_argument("--frames_sampled_real",type=int,default=1) 
    p.add_argument("--balanced",type=int,default=1) 
    p.add_argument('--detector_config', type=str,
                    default='/sotossta/DecepTIV/classification/configs/detectors/efficientnetb4.yaml',
                    help='path to detector YAML file')
    args = p.parse_args()
    print(args)
    main(args)

    



   