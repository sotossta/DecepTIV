import argparse
import json
from utils import load_generator_from_config
import yaml
from tqdm import tqdm

def main(data_dir: str,
         category: str,
         prompt_file: str,
         generator_config: str,
         ):
    # load config
    with open(args.generator_config, 'r') as f:
        config = yaml.safe_load(f)
    config = config.copy()
    #Import text prompts
    with open(prompt_file, 'r') as f:
        prompts = json.load(f)
    video_names = list(prompts.keys())
    #Load generator
    generator = load_generator_from_config(config)
    #Loop over videos
    for video_name in tqdm(video_names):

        prompt = prompts.get(video_name,"")
        generator.generate_and_save_fake_video(prompt = prompt,
                                                            data_dir = data_dir,
                                                            category = category,
                                                            video_name = video_name)
if __name__ == "__main__":

    p = argparse.ArgumentParser(description="Generate_fake_videos")
    p.add_argument("--data_dir", type=str, default="/sotossta/DecepTIV/Dataset")
    p.add_argument("--category", type=str,choices =["Firefighter", "Weather", "Soldier"], default="Firefighter")
    p.add_argument("--prompt_file", type=str, required=True)
    p.add_argument("--generator_config", type=str,
                    default='/sotossta/DecepTIV/generate_fake_videos/configs/CogVideo.yaml',
                    help='path to generator YAML file')
    args = p.parse_args()   
    print(args)
    main(data_dir = args.data_dir, category = args.category, prompt_file = args.prompt_file, generator_config = args.generator_config)