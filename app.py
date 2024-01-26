import os
import gradio as gr
from scripts.text2img import t2i
import scripts.safetensors_to_diffusers as std
import scripts.image_convertions as img_conv

text2img = t2i()
images  = ""
seeds = ""
info = ""
def models():
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

    return converted_paths, safetensor_paths
        
converted, safetensor = models()

def generate_image(selected_model, prompt, negative_prompt, clip_skip, batch_size, num_inference_steps, guidance_scale, width, height):
    print(selected_model)
    images, seeds, info = text2img.generate_image(
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
    img_conv.save_image(images, seeds, info)
    return images

def convert_model(safetensor_model):
    print(safetensor_model)
    std.convert(model_path=safetensor_model)


with gr.Blocks() as interface:
    with gr.Tabs():
        with gr.TabItem("Text2Img"):
            with gr.Blocks():
                selected_model = gr.Dropdown(converted, value=converted[0], label="Models", info="You can select desired models here")

            with gr.Group():
                with gr.Row():
                    prompt = gr.Textbox(placeholder="Prompt Here...",show_label=False)
                    generate_button = gr.Button(value="Generate")
                negative_prompt = gr.Textbox(placeholder="Negative Prompt...",show_label=False)

            with gr.Blocks() as settings: 
                with gr.Row():   
                    with gr.Group():  
                        width = gr.Slider(minimum=512, maximum=1920, label="Width")
                        height = gr.Slider(minimum=512, maximum=1080, label="Height") 
                        clip_skip = gr.Slider(minimum=1, maximum=4, label="Click Skip")
                        batch_size = gr.Slider(minimum=1, maximum=50, label="Batch Size")
                        num_inference_steps = gr.Slider(minimum=1, value=20, label="Num Inference Steps")
                        guidance_scale = gr.Slider(minimum=1, maximum=10, value=7.5 ,label="Guidance Scale")
                    images = gr.Gallery(label="Generated images", show_label=False, columns=[3], rows=[1], object_fit="contain", height="auto")

        with gr.TabItem("Model Converter"):
            with gr.Blocks():
                safetensor_model = gr.Dropdown(safetensor, label="Safetensors", info="You can select a model to turn diffusers")
                convert_button = gr.Button(value="Convert")
    
    generate_button.click(generate_image, inputs=[selected_model, prompt, negative_prompt, clip_skip, batch_size, 
                               num_inference_steps, guidance_scale, width, height], outputs=images)
    convert_button.click(convert_model, inputs=safetensor_model)
    live=True

interface.launch()
