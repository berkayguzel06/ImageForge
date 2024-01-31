import os
import gradio as gr
from scripts.text2img import t2i
from scripts.img2img import i2i
import scripts.safetensors_to_diffusers as std
import scripts.image_convertions as img_conv

text2img = t2i()
img2img = i2i()
images  = ""
seeds = ""
info = ""

def get_selections():
    converted_models_dir = "convertedModels"
    safetensor_dir = "model"
    stable_dif_model = "runwayml/stable-diffusion-v1-5"

    safetensors = os.listdir(safetensor_dir)
    converted_models = os.listdir(converted_models_dir)
    converted_models.remove("convertedmodels.txt")
    safetensors.remove("convertedmodels.txt")

    safetensor_paths = [os.path.join(safetensor_dir, model) for model in safetensors]

    converted_paths = []
    converted_paths.append(stable_dif_model)
    for model in converted_models:
        path = os.path.join(converted_models_dir, model)
        converted_paths.append(path)
    
    schedulers = get_schedulers()

    return converted_paths, safetensor_paths, schedulers

def get_schedulers():
    extract = []
    schedulers = text2img.pipe.get_schedulers()
    for sch in schedulers:
        scheduler = str(sch).split("'")[1]
        schdeuler = scheduler.split(".")[3]
        extract.append(schdeuler)
    print(extract)
    return extract

def find_schedulers(scheduler):
    schedulers = text2img.pipe.get_schedulers()
    for idx in range(len(schedulers)):
        result = str(schedulers[idx]).find(scheduler)
        if result != -1:
            print(schedulers[idx])
            return schedulers[idx]
    return None

converted, safetensor, schedulers = get_selections()

def generate_image(scheduler, selected_model, prompt, negative_prompt, clip_skip, batch_size, num_inference_steps, guidance_scale, width, height):
    print(selected_model)
    sch = find_schedulers(scheduler)
    images, seeds, info = text2img.generate_image(
        scheduler=sch,
        selected_model=selected_model,
        prompt=prompt, 
        negative_prompt=negative_prompt, 
        clip_skip=clip_skip, 
        batch_size=batch_size, 
        num_inference_steps=num_inference_steps, 
        guidance_scale=guidance_scale, 
        width=width, 
        height=height
    )
    img_conv.save_image(images, seeds, info, "t2i")
    return images

def i2i_generate_image(scheduler, selected_model, prompt, negative_prompt, clip_skip, batch_size, num_inference_steps, guidance_scale, width, height, strength, image):
    print(selected_model)
    picture = img_conv.change_size(image, width, height)
    sch = find_schedulers(scheduler)
    images, seeds, info = img2img.generate_image(
        scheduler=sch,
        image=picture,
        selected_model=selected_model,
        prompt=prompt, 
        negative_prompt=negative_prompt, 
        clip_skip=clip_skip, 
        batch_size=batch_size, 
        num_inference_steps=num_inference_steps, 
        guidance_scale=guidance_scale, 
        width=width, 
        height=height,
        strength=strength
    )
    img_conv.save_image(images, seeds, info, "i2i")
    return images

def convert_model(safetensor_model):
    print(safetensor_model)
    std.convert(model_path=safetensor_model)

with gr.Blocks(title="ImageForge") as interface:
    with gr.Blocks():
        selected_model = gr.Dropdown(converted, value=converted[0], label="Models", info="You can select desired models here")
        scheduler = gr.Dropdown(schedulers, value=schedulers[0], label="Schedulers")
    with gr.Tabs():
        with gr.TabItem("Text2Img"):
            with gr.Group():
                with gr.Row():
                    prompt = gr.Textbox(placeholder="Prompt Here...",show_label=False)
                    generate_button = gr.Button(value="Generate")
                negative_prompt = gr.Textbox(placeholder="Negative Prompt...",show_label=False)

            with gr.Blocks(): 
                with gr.Row():   
                    with gr.Group():  
                        width = gr.Slider(minimum=512, maximum=1920, label="Width")
                        height = gr.Slider(minimum=512, maximum=1080, label="Height") 
                        clip_skip = gr.Slider(minimum=1, maximum=4, label="Click Skip")
                        batch_size = gr.Slider(minimum=1, maximum=50, label="Batch Size")
                        num_inference_steps = gr.Slider(minimum=1, value=20, label="Num Inference Steps")
                        guidance_scale = gr.Slider(minimum=1, maximum=10, value=7.5 ,label="Guidance Scale")
                    images = gr.Gallery(label="Generated images", show_label=False, columns=[3], rows=[1], object_fit="contain", height="auto")
            generate_button.click(generate_image, inputs=[scheduler, selected_model, prompt, negative_prompt, clip_skip, batch_size, 
                               num_inference_steps, guidance_scale, width, height], outputs=images)
        
        with gr.TabItem("Img2Img"):
            with gr.Group():
                with gr.Row():
                    i2i_prompt = gr.Textbox(placeholder="Prompt Here...",show_label=False)
                    i2i_generate_button = gr.Button(value="Generate")
                i2i_negative_prompt = gr.Textbox(placeholder="Negative Prompt...",show_label=False)

            with gr.Blocks(): 
                with gr.Row():   
                    with gr.Group():  
                        img = gr.Image(type="pil", height="30vw")
                        i2i_width = gr.Slider(minimum=512, maximum=1920, label="Width")
                        i2i_height = gr.Slider(minimum=512, maximum=1080, label="Height") 
                        i2i_clip_skip = gr.Slider(minimum=1, maximum=4, label="Click Skip")
                        i2i_batch_size = gr.Slider(minimum=1, maximum=50, label="Batch Size")
                        i2i_num_inference_steps = gr.Slider(minimum=1, value=20, label="Num Inference Steps")
                        i2i_guidance_scale = gr.Slider(minimum=1, maximum=10, value=7.5 ,label="Guidance Scale")
                        i2i_strength = gr.Slider(minimum=0, value=0.8, maximum=1, label="Strength")
                    i2i_images = gr.Gallery(label="Generated images", show_label=False, columns=[3], rows=[1], object_fit="contain", height="auto")
            i2i_generate_button.click(i2i_generate_image, inputs=[scheduler, selected_model, i2i_prompt, i2i_negative_prompt, i2i_clip_skip, i2i_batch_size, 
                               i2i_num_inference_steps, i2i_guidance_scale, i2i_width, i2i_height, i2i_strength, img], outputs=i2i_images)
    
        with gr.TabItem("Model Converter"):
            with gr.Blocks():
                safetensor_model = gr.Dropdown(safetensor, label="Safetensors", info="You can select a model to turn diffusers")
                convert_button = gr.Button(value="Convert")
    
    convert_button.click(convert_model, inputs=safetensor_model)

    live=True

interface.launch(inbrowser=True)
