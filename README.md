# ImageForge
## Text2Img
![t2i_1](https://github.com/berkayguzel06/ImageForge/assets/98205992/58a63d86-7a1f-461c-95fa-439649b6a63f)

## Img2Img
![i2i_1](https://github.com/berkayguzel06/ImageForge/assets/98205992/a4ff3692-6c2e-4b67-9857-062fc173dda3)

## PNG info extract
![png_info_1](https://github.com/berkayguzel06/ImageForge/assets/98205992/1f53fdbd-3b6f-4bd9-950d-d639534bf0fe)

## Features

- Supports CUDA devices for accelerated processing.
- txt2img and img2img mode.
- Supports for single LoRA.
- Supports for different scheduler types.
- Extract PNG image info from generated images.
- One click install and run script (but you still must install python and git)
- Utilizes Hugging Face Diffusers and Transformers libraries for AI models.
- Employs the Gradio library for the user interface.
- Users can easily change image models.
- Supports uploading new `.safetensors` models (not directly use).
- Provides a conversion mechanism to convert `.safetensors` models into the required Diffusers model format.
- Changeable image settings (width, height, number of inference step, guidance scale)
- Supports multiple image generation with batch size

## Prerequisites

- CUDA-enabled device (if using GPU acceleration).
- Python 3.x
- Dependencies specified in `requirements.txt`.

## Installation on Windows
1. Install Python 3.10.6 (Newer version of Python does not support torch), checking "Add Python to PATH".
2. Install git.
3. Download the ImageForge repository, for example by running git clone https://github.com/berkayguzel06/ImageForge.git.
4. Run start-browser.bat from Windows Explorer as normal, non-administrator, user.
