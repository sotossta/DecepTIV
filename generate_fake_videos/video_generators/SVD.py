"""
Reference:
@article{blattmann2023stable,
  title={Stable video diffusion: Scaling latent video diffusion models to large datasets},
  author={Blattmann, Andreas and Dockhorn, Tim and Kulal, Sumith and Mendelevitch, Daniel and Kilian, Maciej and Lorenz, Dominik and Levi, Yam and English, Zion and Voleti, Vikram and Letts, Adam and others},
  journal={arXiv preprint arXiv:2311.15127},
  year={2023}
}

Github code: https://github.com/Stability-AI/generative-models
"""

import torch
from diffusers import StableVideoDiffusionPipeline
from diffusers.utils import export_to_video, load_image
import os
from utils import extract_first_frame
# SVD Generator class definition
class SVD_Generator(object):

    def __init__(self, config):
        super(SVD_Generator, self).__init__()

        self.config = config
        #Build generator
        self.generator = self.build_generator()

    def build_generator(self):
        
        pipe = StableVideoDiffusionPipeline.from_pretrained("stabilityai/stable-video-diffusion-img2vid-xt",
                                                            torch_dtype=torch.float16, variant="fp16").to("cuda")  
        pipe.set_progress_bar_config(disable=True) # Disabled progress bar for diffusion steps, you can comment this out to enable it back again
        if self.config['cpu_offload']==True:
            pipe.enable_sequential_cpu_offload()# cpu offloading significantly decreases VRAM usage but slows generation down
        return pipe

    def generate_and_save_fake_video(self, prompt, data_dir, category, video_name):

        img_pil = extract_first_frame(os.path.join(data_dir,"Real",category,"videos",video_name))
        image = load_image(image = img_pil)
        image = image.resize((1024, 576))
        video = self.generator(image,
                                num_inference_steps=self.config['inference_steps'],
                                decode_chunk_size = self.config['decode_chunk_size'],
                                num_frames = self.config['n_frames']).frames[0]
        print("torch.cuda.max_memory_reserved: %fGB"%(torch.cuda.max_memory_reserved(0)/1024/1024/1024),flush=True) # print peak VRAM usage
        save_dir = os.path.join(data_dir,self.config['generator'],category,"videos")
        os.makedirs(save_dir, exist_ok=True)
        export_to_video(video, os.path.join(save_dir, video_name), fps=self.config['save_fps'])
