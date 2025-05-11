"""
Reference:
@misc{Luma,
  title = {Luma Dream Machine},
  howpublished = {https://lumalabs.ai/dream-machine},
  note = {Accessed: 2025-14-04}
}

Github API code: https://github.com/lumalabs/lumaai-python
"""

from lumaai import LumaAI
import requests
import os
from utils import extract_first_frame
from openai import OpenAI
# Luma Generator class definition
class Luma_Generator(object):

    def __init__(self, config):
        super(Luma_Generator, self).__init__()

        self.config = config
        #Build generator
        self.generator = self.build_generator()

    def build_generator(self):
        
        #Setup Luma
        with open(os.path.join(self.config['luma_key_dir']), "r") as file:
            api_key = file.read().strip()
        client = LumaAI(auth_token=api_key)
        return client

    def generate_and_save_fake_video(self, prompt, data_dir, category, video_name):
        
        img_pil = extract_first_frame(os.path.join(data_dir,"Real",category,"videos",video_name))
        if self.config['prompt_enhancement']==True:
                prompt = self.prompt_enhancement(original_prompt = prompt, image=img_pil)
        task = self.generator.generations.create(
          model= self.config['model_version'],
          prompt=prompt
          )
        while True:
            task = self.generator.generations.get(id=task.id)
            if task.state == "completed":
                 break
        video_url = task.assets.video
        response = requests.get(video_url, stream=True)
        save_dir = os.path.join(data_dir,self.config['generator'],category,"videos")
        os.makedirs(save_dir, exist_ok=True)
        with open(os.path.join(save_dir,video_name), 'wb') as file:
            file.write(response.content)
        print(f"Generated: {video_name}")

    def prompt_enhancement(self, original_prompt, image=None):

        """
        Adapted from: https://lumalabs.ai/learning-hub/best-practices
        """

        with open(self.config['openai_key_dir'], 'r') as f:
            key= f.read().strip()
            client = OpenAI(api_key=key)

        sys_prompt = """You are part of a team of bots whose job is to create videos given a descriptive prompt. You will be
        given an initial text prompt that describes a video, your job is to refine that prompt such that it follows some specific rules.
        The quality of the generated video will highly depend on how well the refined prompt follows the specified rules. The rules
        are to follow are:
        
        1) Describe what you want in natural, detailed language
        2) Be specific about the style, mood, lighting, or elements you want to see. Adding adjectives and clear descriptors helps the model generate more accurate and tailored results. 
        3) Add cinematic movement to videos with options like Pan, Orbit, or Zoom.
        """
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
            {"role": "system", "content": f"{sys_prompt}"},
            {"role": "user", "content": f"{original_prompt}"},
            ],
            temperature=0.01,
            top_p=0.7,
            stream=False,
            max_tokens=250,
            )
        enhanced_prompt = response.choices[0].message.content
        return enhanced_prompt