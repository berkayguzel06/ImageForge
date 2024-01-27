# ImageForge
![image](https://github.com/berkayguzel06/ImageForge/assets/98205992/63136c33-d867-4553-8d5f-dbf9d0aeeb76)

## Features

- Supports CUDA devices for accelerated processing.
- Original txt2img mode.
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
