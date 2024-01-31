import diffusers
import transformers
import random
import torch
import matplotlib.pyplot as plt
import numpy as np
from scripts.pipe_engine import pipeline

class i2i:
    def __init__(self):
        self.pipe = pipeline()
        self.device_name = self.pipe.device_name
        self.torch_dtype = self.pipe.torch_dtype

    def generate_image(self, lora, lora_weight, scheduler, image, selected_model, prompt, batch_size, num_inference_steps, guidance_scale, negative_prompt, width, height, clip_skip, strength):
            pipe_type = "i2i"
            
            self.pipe.system_check(pipe_type, clip_skip, selected_model, scheduler, lora)
            diffuser_pipe = self.pipe.get_pipe()

            prompt_embeds, negative_prompt_embeds = self.get_prompt_embeddings(diffuser_pipe,prompt,negative_prompt,device = self.device_name)
            use_prompt_embeddings = False
            start_idx = 0
            seeds = [random.randint(0,99999) for i in range(start_idx , start_idx + batch_size, 1)]

            batch_size = batch_size
            num_inference_steps = num_inference_steps
            guidance_scale = guidance_scale
            width  = width
            height = height
            sm="runwayml/stable-diffusion-v1-5"
            if selected_model!=sm:
                sm = selected_model.split("\\")[1]
                sm = sm.split(".")[0]
            else:
                sm = selected_model.split("/")[1]

            sch = str(scheduler).split("'")[1]
            sch = sch.split(".")[3]
            
            info = {
                "model":sm,
                "scheduler":sch,
                "prompt":prompt,
                "negative_prompt":negative_prompt,
                "num_inference_steps":num_inference_steps,
                "guidance_scale":guidance_scale,
                "width":width,
                "height":height,
                "strength":strength
            }
            images = []
            for count, seed in enumerate(seeds):
                if use_prompt_embeddings is False:
                    new_img = diffuser_pipe(
                        image=image,
                        prompt = prompt,
                        negative_prompt = negative_prompt,
                        width = width,
                        height = height,
                        guidance_scale = guidance_scale,
                        num_inference_steps = num_inference_steps,
                        num_images_per_prompt = 1,
                        strength=strength,
                        cross_attention_kwargs={"scale":lora_weight},
                        generator = torch.manual_seed(seed),
                    ).images
                else:
                    new_img = diffuser_pipe(
                        image=image,
                        prompt_embeds = prompt_embeds,
                        negative_prompt_embeds = negative_prompt_embeds,
                        width = width,
                        height = height,
                        guidance_scale = guidance_scale,
                        num_inference_steps = num_inference_steps,
                        num_images_per_prompt = 1,
                        strength=strength,
                        generator = torch.manual_seed(seed),
                    ).images

                images = images + new_img

            return images, seeds, info

    def get_prompt_embeddings(self, pipe, prompt, negative_prompt, split_character = ",", device = torch.device("cpu")):
        max_length = pipe.tokenizer.model_max_length
        count_prompt = len(prompt.split(split_character))
        count_negative_prompt = len(negative_prompt.split(split_character))

        if count_prompt >= count_negative_prompt:
            input_ids = pipe.tokenizer(prompt, return_tensors = "pt", truncation = False).input_ids.to(device)
            shape_max_length = input_ids.shape[-1]
            negative_ids = pipe.tokenizer(negative_prompt,truncation = False,padding = "max_length",
                                          max_length = shape_max_length,return_tensors = "pt").input_ids.to(device)
        else:
            negative_ids = pipe.tokenizer(negative_prompt, return_tensors = "pt", truncation = False).input_ids.to(device)
            shape_max_length = negative_ids.shape[-1]
            input_ids = pipe.tokenizer(prompt,return_tensors = "pt",truncation = False,padding = "max_length",
                                       max_length = shape_max_length).input_ids.to(device)

        concat_embeds = []
        neg_embeds = []
        for i in range(0, shape_max_length, max_length):
            concat_embeds.append(
                pipe.text_encoder(input_ids[:, i: i + max_length])[0]
            )
            neg_embeds.append(
                pipe.text_encoder(negative_ids[:, i: i + max_length])[0]
            )

        return torch.cat(concat_embeds, dim = 1), torch.cat(neg_embeds, dim = 1)