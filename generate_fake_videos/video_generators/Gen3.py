"""
Reference:
@misc{Runway-Gen3,
  title = {Introducing Gen-3 Alpha: A New Frontier for Video Generation.},
  howpublished = {https://runwayml.com/research/introducing-gen-3-alpha/},
  note = {Accessed: 2025-14-04}
}

Github API code: https://github.com/runwayml/sdk-python
"""

from runwayml import RunwayML
import requests
import os
from utils import extract_first_frame, pil_image_to_url
from openai import OpenAI
# Gen3 Generator class definition
class Gen3_Generator(object):

    def __init__(self, config):
        super(Gen3_Generator, self).__init__()

        self.config = config
        #Build generator
        self.generator = self.build_generator()

    def build_generator(self):
        
        #Setup GEN3
        with open(os.path.join(self.config['runwayml_key_dir']), "r") as file:
            api_key = file.read().strip()
        client = RunwayML(api_key = api_key)
        return client

    def generate_and_save_fake_video(self, prompt, data_dir, category, video_name):
        
        img_pil = extract_first_frame(os.path.join(data_dir,"Real",category,"videos",video_name))
        if self.config['prompt_enhancement']==True:
                prompt = self.prompt_enhancement(original_prompt = prompt, image=img_pil)
        task = self.generator.image_to_video.create(
          model= self.config['model_version'],
          prompt_image=pil_image_to_url(img_pil),
          duration =self.config['duration'],
          ratio=self.config['ratio'],
          prompt_text=prompt
          )
        while True:
            if self.generator.tasks.retrieve(task.id).status=="SUCCEEDED":
                break    
        video_url = self.generator.tasks.retrieve(task.id).output[0]
        response = requests.get(video_url, stream=True)
        save_dir = os.path.join(data_dir,self.config['generator'],category,"videos")
        os.makedirs(save_dir, exist_ok=True)
        with open(os.path.join(save_dir,video_name), 'wb') as file:
            file.write(response.content)
        print(f"Generated: {video_name}")

    def prompt_enhancement(self, original_prompt, image=None):

        """
        Adapted from: https://help.runwayml.com/hc/en-us/articles/30586818553107-Gen-3-Alpha-Prompting-Guide
        """

        with open(self.config['openai_key_dir'], 'r') as f:
            key= f.read().strip()
            client = OpenAI(api_key=key)

        sys_prompt = """You are part of a team of bots whose job is to create videos giving a starting frame and a descriptive prompt.
        When you are given both an image and a prompt you should refactor the prompt such that it is align with the following rules.
        Creating a strong prompt that conveys the scene is the key to generating video aligned with the concept the rules are the follwoing.
        
        1) All prompts should be direct and easily understood, not conceptual. Avoid using overly conceptual language and phrasing when 
        a simplistic description would efficiently convey the scene. 
        
        2) Prompts should be descriptive, not conversational or command-based
        
        3) Prompts should use positive phrasing. Negative prompts, or prompts that describe what shouldn't happen, should not be included
        
        4) Create a simple and direct text prompt that describes the movement you'd like in the output. You do not need to describe the contents of the image.
        
        5) Keywords can be beneficial to achieve specific styles in your output. Ensuring that keywords are cohesive with your overall prompt will 
        make them more apparent in your output. In example, including keywords about skin texture wouldn't be beneficial to a wide angle shot where 
        the camera is not closely focused on a face. A wide angle shot might instead benefit from additional details about the environment
        
        Lastly, the output should be only the enhanced text prompt following the above rules. If the given image and the description do not match, 
        generate your own prompt according to these rules but only following the image.
        """
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[ 
                {"role": "system", "content": f"{sys_prompt}"},
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