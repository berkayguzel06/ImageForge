import diffusers
import transformers
import random
import torch
import matplotlib.pyplot as plt
import numpy as np

class t2i:
    def __init__(self):
        self.device_name = ""
        self.torch_dtype = ""
        self.model_path = "runwayml/stable-diffusion-v1-5"
        self.cache_dir = ".cache"
        self.clip_skip = 1
        self.cuda()
        self.setup_system()

    def cuda(self):
        if torch.cuda.is_available():
            print("Cuda is available")
            self.device_name = torch.device("cuda")
            self.torch_dtype = torch.float16
        else:
            self.device_name = torch.device("cpu")
            self.torch_dtype = torch.float32

    def setup_system(self):

        model_dir = None

        if self.clip_skip > 1:
            self.text_encoder = transformers.CLIPTextModel.from_pretrained("runwayml/stable-diffusion-v1-5", cache_dir = self.cache_dir, subfolder = "text_encoder",
                                                                      num_hidden_layers = 12 - (self.clip_skip - 1),torch_dtype = self.torch_dtype)
        else:
            self.text_encoder = transformers.CLIPTextModel.from_pretrained("runwayml/stable-diffusion-v1-5",cache_dir = self.cache_dir,subfolder = "text_encoder",num_hidden_layers = 12,torch_dtype = self.torch_dtype)

        if self.model_path == "runwayml/stable-diffusion-v1-5":
            model_dir = self.cache_dir
        else:
            model_dir = None

        if self.clip_skip > 1:
            self.pipe = diffusers.DiffusionPipeline.from_pretrained(
                self.model_path,
                cache_dir=model_dir,
                torch_dtype = self.torch_dtype,
                safety_checker = None,
                text_encoder = self.text_encoder,
            )
        else:
            self.pipe = diffusers.DiffusionPipeline.from_pretrained(
                self.model_path,
                cache_dir=model_dir,
                torch_dtype = self.torch_dtype,
                safety_checker = None
            )

    def system_check(self,clip_skip,selected_model):
            
            if self.clip_skip!=clip_skip:
                self.clip_skip = clip_skip
                self.setup_system()

            if selected_model!=self.model_path:
                self.model_path = selected_model
                self.setup_system()

    def generate_image(self, selected_model, prompt, batch_size, num_inference_steps, guidance_scale, negative_prompt, width, height, clip_skip):
            
            self.system_check(clip_skip,selected_model)
            self.pipe = self.pipe.to(self.device_name)
            self.pipe.scheduler = diffusers.EulerAncestralDiscreteScheduler.from_config(
                self.pipe.scheduler.config
            )

            prompt_embeds, negative_prompt_embeds = self.get_prompt_embeddings(self.pipe,prompt,negative_prompt,device = self.device_name)
            use_prompt_embeddings = True
            start_idx = 0
            seeds = [random.randint(0,99999) for i in range(start_idx , start_idx + batch_size, 1)]

            batch_size = batch_size
            num_inference_steps = num_inference_steps
            guidance_scale = guidance_scale
            width  = width
            height = height
            info = {
                "prompt":prompt,
                "negative_prompt":negative_prompt,
                "num_inference_steps":num_inference_steps,
                "guidance_scale":guidance_scale,
                "width":width,
                "height":height
            }
            images = []
            for count, seed in enumerate(seeds):
                if use_prompt_embeddings is False:
                    new_img = self.pipe(
                        prompt = prompt,
                        negative_prompt = negative_prompt,
                        width = width,
                        height = height,
                        guidance_scale = guidance_scale,
                        num_inference_steps = num_inference_steps,
                        num_images_per_prompt = 1,
                        generator = torch.manual_seed(seed),
                    ).images
                else:
                    new_img = self.pipe(
                        prompt_embeds = prompt_embeds,
                        negative_prompt_embeds = negative_prompt_embeds,
                        width = width,
                        height = height,
                        guidance_scale = guidance_scale,
                        num_inference_steps = num_inference_steps,
                        num_images_per_prompt = 1,
                        generator = torch.manual_seed(seed),
                    ).images

                images = images + new_img

            return images, seeds, info

    def get_prompt_embeddings(self, pipe,prompt,negative_prompt,split_character = ",",device = torch.device("cpu")):
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