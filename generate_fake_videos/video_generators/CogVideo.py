"""
Reference:
@article{hong2022cogvideo,
  title={Cogvideo: Large-scale pretraining for text-to-video generation via transformers},
  author={Hong, Wenyi and Ding, Ming and Zheng, Wendi and Liu, Xinghan and Tang, Jie},
  journal={arXiv preprint arXiv:2205.15868},
  year={2022}
}

Github code: https://github.com/THUDM/CogVideo/
"""

import torch
from diffusers import CogVideoXPipeline, CogVideoXImageToVideoPipeline
from diffusers.utils import export_to_video, load_image
import os
from utils import extract_first_frame, pil_image_to_url
from openai import OpenAI
# CogVideo Generator class definition
class CogVideo_Generator(object):

    def __init__(self, config):
        super(CogVideo_Generator, self).__init__()

        self.config = config
        #Build generator
        self.generator = self.build_generator()

    def build_generator(self):
        
        if self.config['mode']=="T2V":
            pipe = CogVideoXPipeline.from_pretrained("THUDM/CogVideoX-5b",torch_dtype=torch.float16).to("cuda")  
        else:
            pipe = CogVideoXImageToVideoPipeline.from_pretrained("THUDM/CogVideoX-5b-I2V",torch_dtype=torch.float16).to("cuda")

        pipe.vae.enable_slicing()
        pipe.vae.enable_tiling()
        pipe.set_progress_bar_config(disable=True) # Disabled progress bar for diffusion steps, you can comment this out to enable it back again
        if self.config['cpu_offload']==True:
            pipe.enable_sequential_cpu_offload()# cpu offloading significantly decreases VRAM usage but slows generation down
        return pipe


    def generate_and_save_fake_video(self, prompt, data_dir, category, video_name):

        if self.config['mode']=="T2V":
    
            if self.config['prompt_enhancement']==True:
                prompt = self.prompt_enhancement(original_prompt = prompt)

            video = self.generator(
                    prompt=prompt,
                    num_videos_per_prompt=1,
                    num_inference_steps=self.config['inference_steps'],
                    num_frames=self.config['n_frames'],
                    guidance_scale=self.config['guidance_scale'],
                ).frames[0]
        else:
            img_pil = extract_first_frame(os.path.join(data_dir,"Real",category,"videos",video_name))
            if self.config['prompt_enhancement']==True:
                prompt = self.prompt_enhancement(original_prompt = prompt, image =img_pil)
            image = load_image(image = img_pil)
            video = self.generator(
                prompt=prompt,
                image = image,
                num_videos_per_prompt=1,
                num_inference_steps=self.config['inference_steps'],
                num_frames=self.config['n_frames'],
                guidance_scale=self.config['guidance_scale'],
            ).frames[0]
        print("torch.cuda.max_memory_reserved: %fGB"%(torch.cuda.max_memory_reserved(0)/1024/1024/1024),flush=True) # print peak VRAM usage
        save_dir = os.path.join(data_dir,f"{self.config['generator']}_{self.config['mode']}",category,"videos")
        os.makedirs(save_dir, exist_ok=True)
        export_to_video(video, os.path.join(save_dir, video_name), fps=self.config['save_fps'])

    def prompt_enhancement(self, original_prompt, image=None):

        """
        Adapted from: https://github.com/THUDM/CogVideo/blob/main/inference/convert_demo.py
        """

        with open(self.config['openai_key_dir'], 'r') as f:
            key= f.read().strip()
            client = OpenAI(api_key=key)
        sys_prompt_t2v = """You are part of a team of bots that creates videos. You work with an assistant bot that will draw anything you say in square brackets.

        For example , outputting " a beautiful morning in the woods with the sun peaking through the trees " will trigger your partner bot 
        to output an video of a forest morning , as described. You will be prompted by people looking to create detailed , amazing videos. 
        The way to accomplish this is to take their short prompts and make them extremely detailed and descriptive.
        There are a few rules to follow:

        You will only ever output a single video description per user request.

        When modifications are requested , you should not simply make the description longer . You should refactor the entire description to integrate the suggestions.
        Other times the user will not want modifications , but instead want a new image . In this case , you should ignore your previous conversation with the user.

        Video descriptions must have the same num of words as examples below. Extra words will be ignored.
        """

        sys_prompt_i2v = """
        **Objective**: **Give a highly descriptive video caption based on input image and user input. **. As an expert, delve deep into the image with a discerning eye, 
        leveraging rich creativity, meticulous thought. When describing the details of an image, include appropriate dynamic information to ensure that the video caption contains 
        reasonable actions and plots. If user input is not empty, then the caption should be expanded according to the user's input. 

        **Note**: The input image is the first frame of the video, and the output video caption should describe the motion starting from the current image. 
        User input is optional and can be empty. 

        **Note**: Don't contain camera transitions!!! Don't contain screen switching!!! Don't contain perspective shifts !!!

        **Answering Style**:
        Answers should be comprehensive, conversational, and use complete sentences. The answer should be in English no matter what the user's input is. 
        Provide context where necessary and maintain a certain tone.  Begin directly without introductory phrases like "The image/video showcases" "The photo captures" and more.
        For example, say "A woman is on a beach", instead of "A woman is depicted in the image".

        **Output Format**: "[highly descriptive image caption here]"

        user input:
        """
        if self.config['mode'] =="T2V":
            response = client.chat.completions.create(
                model="gpt-4o",
                    messages=[
                            {"role": "system", "content": f"{sys_prompt_t2v}"},
                            {
                                "role": "user",
                                "content": 'Create an imaginative video descriptive caption or modify an earlier caption for the user input : " a girl is on the beach"',
                            },
                            {
                                "role": "assistant",
                                "content": "A radiant woman stands on a deserted beach, arms outstretched, wearing a beige trench coat, white blouse, light blue jeans, and chic boots, against a backdrop of soft sky and sea. Moments later, she is seen mid-twirl, arms exuberant, with the lighting suggesting dawn or dusk. Then, she runs along the beach, her attire complemented by an off-white scarf and black ankle boots, the tranquil sea behind her. Finally, she holds a paper airplane, her pose reflecting joy and freedom, with the ocean's gentle waves and the sky's soft pastel hues enhancing the serene ambiance.",
                            },
                            {
                                "role": "user",
                                "content": 'Create an imaginative video descriptive caption or modify an earlier caption for the user input : " A man jogging on a football field"',
                            },
                            {
                                "role": "assistant",
                                "content": "A determined man in athletic attire, including a blue long-sleeve shirt, black shorts, and blue socks, jogs around a snow-covered soccer field, showcasing his solitary exercise in a quiet, overcast setting. His long dreadlocks, focused expression, and the serene winter backdrop highlight his dedication to fitness. As he moves, his attire, consisting of a blue sports sweatshirt, black athletic pants, gloves, and sneakers, grips the snowy ground. He is seen running past a chain-link fence enclosing the playground area, with a basketball hoop and children's slide, suggesting a moment of solitary exercise amidst the empty field.",
                            },
                            {
                                "role": "user",
                                "content": 'Create an imaginative video descriptive caption or modify an earlier caption for the user input : " A woman is dancing, HD footage, close-up"',
                            },
                            {
                                "role": "assistant",
                                "content": "A young woman with her hair in an updo and wearing a teal hoodie stands against a light backdrop, initially looking over her shoulder with a contemplative expression. She then confidently makes a subtle dance move, suggesting rhythm and movement. Next, she appears poised and focused, looking directly at the camera. Her expression shifts to one of introspection as she gazes downward slightly. Finally, she dances with confidence, her left hand over her heart, symbolizing a poignant moment, all while dressed in the same teal hoodie against a plain, light-colored background.",
                            },
                            {
                                "role": "user",
                                "content": f'Create an imaginative video descriptive caption or modify an earlier caption in ENGLISH for the user input: " {original_prompt} "',
                            },
                        ],
                    temperature=0.01,
                    top_p=0.7,
                    stream=False,
                    max_tokens=250,
                )
        else:
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[ 
                    {"role": "system", "content": f"{sys_prompt_i2v}"},
                    {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": original_prompt},
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": pil_image_to_url(image),
                                    },
                                },
                            ],
                    },
                ],
                temperature=0.01,
                top_p=0.7,
                stream=False,
                max_tokens=250,
                )
        enhanced_prompt = response.choices[0].message.content
        return enhanced_prompt




