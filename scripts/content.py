import os

def get_selections():
    converted_models_dir = "convertedModels"
    safetensor_dir = "model"
    stable_dif_model = "runwayml/stable-diffusion-v1-5"
    lora_models_dir = "lora_models"

    safetensors = os.listdir(safetensor_dir)
    converted_models = os.listdir(converted_models_dir)
    lora_models = os.listdir(lora_models_dir)

    converted_models.remove("convertedmodels.txt")
    safetensors.remove("convertedmodels.txt")
    lora_models.remove("lora_models.txt")

    safetensor_paths = [os.path.join(safetensor_dir, model) for model in safetensors]

    converted_paths = []
    converted_paths.append(stable_dif_model)
    for model in converted_models:
        path = os.path.join(converted_models_dir, model)
        converted_paths.append(path)

    lora_paths = []
    lora_paths.append(" ")
    for model in lora_models:
        path = os.path.join(lora_models_dir, model)
        lora_paths.append(path)
    
    return converted_paths, safetensor_paths, lora_paths

def get_schedulers(schedulers):
    extract = []
    for sch in schedulers:
        scheduler = str(sch).split("'")[1]
        schdeuler = scheduler.split(".")[3]
        extract.append(schdeuler)
    print(extract)
    return extract