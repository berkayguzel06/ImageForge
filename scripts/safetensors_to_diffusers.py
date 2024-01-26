import diffusers
import torch
import os

model_place = "convertedModels"
extension = ".conv"
cache_dir = ".cache"
device = torch.device("cuda")
dtype = torch.float16

def get_model_name(model_path):
    path = model_path.split("\\")[1]
    model_name = path.split(".")[0]
    return model_name

def convert(model_path):
    model_name = get_model_name(model_path)
    model_name = model_name+extension
    out_path = os.path.join(model_place,model_name)
    print(f"{model_path} converting to diffusers in {out_path} as {model_name}.")
    pipe = diffusers.StableDiffusionPipeline.from_single_file(model_path, cache_dir=cache_dir, torch_dtype = dtype)
    pipe = pipe.to(device)
    pipe.save_pretrained(out_path, safe_serialization=True)
    print("Model created.")