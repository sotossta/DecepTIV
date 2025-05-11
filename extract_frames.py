import os
from os.path import join, exists
import argparse
import cv2
from tqdm import tqdm

def extract_frames(data_path, output_path,max_frames):
    os.makedirs(output_path, exist_ok=True)
    reader = cv2.VideoCapture(data_path)
    frame_num = 0
    
    while reader.isOpened() and frame_num < max_frames:
        success, image = reader.read()
        if not success:
            break
        
        cv2.imwrite(join(output_path, '{:04d}.png'.format(frame_num)), image)
        frame_num += 1
    
    reader.release()

def extract_videos(args):
    datasets = [args.dataset] if args.dataset != "all" else [
        "Real", "CogVideo_T2V", "CogVideo_I2V", "Open-Sora", "SVD", "HunyuanVideo",
        "EasyAnimate_T2V", "EasyAnimate_I2V", "DynamiCrafter", "Wan2.1", "Luma", "Gen3", 
    ]
    categories = ["Firefighter", "Weather", "Soldier"] if args.category == "all" else [args.category]
    video_files = []
    for dataset in datasets:
        for category in categories:

            if args.perturbed==0:

                videos_path = join(args.base_dir, dataset, category, "videos")
                images_path = join(args.base_dir, dataset, category, "images")
            else:
                videos_path = join(args.base_dir, dataset, category, "videos_pert")
                images_path = join(args.base_dir, dataset, category, "images_pert")

            
            if not exists(videos_path):
                print(f"Skipping {dataset}/{category}: No videos found.")
                continue
            for video in os.listdir(videos_path):
                video_name = video.split(".")[0]
                video_output_path = join(images_path, video_name)
                video_files.append((video, videos_path, video_output_path, dataset, category))
    
        
    for video, videos_path, video_output_path, dataset, category in tqdm(video_files, desc="Processing videos"):
        if exists(video_output_path) and os.listdir(video_output_path):
            print(f"Skipping {video} in {dataset}/{category} as frames already exist.", flush=True)
            continue
        
        print(f"Processing {video} in {dataset}/{category}", flush=True)
        extract_frames(join(videos_path, video), video_output_path,args.max_frames)

if __name__ == '__main__':
    p = argparse.ArgumentParser(description="Extract Video Frames")
    p.add_argument("--base_dir", type=str, default="/sotossta/DecepTIV/Dataset",
                   help="The base path of all the datasets")
    p.add_argument("--category", type=str, choices=["Firefighter", "Weather", "Soldier", "all"], default="Firefighter",
                   help="Category of dataset")
    p.add_argument('--dataset', '-d', type=str, choices=
                    ["Real", "CogVideo_T2V", "CogVideo_I2V", "Open-Sora", "SVD", "HunyuanVideo",
                    "EasyAnimate_T2V", "EasyAnimate_I2V", "DynamiCrafter","Wan2.1","Luma", "Gen3","all"], required=True)
    p.add_argument("--max_frames", type=int, default=50, help="Maximum number of frames to extract from each video")
    p.add_argument("--perturbed", type=int, default=0)
    
    args = p.parse_args()
    print(args)
    extract_videos(args)
