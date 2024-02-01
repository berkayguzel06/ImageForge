import os
from PIL import Image
from PIL.PngImagePlugin import PngInfo

def isOpen(image):
    if isinstance(image, Image.Image):
        picture = image
    else:
        picture = Image.open(image)
    return picture

def change_size(image, width, height):
    picture = isOpen(image)
    picture = picture.resize((width, height))
    return picture

def read_png_info(image):
    picture = isOpen(image)
    return picture.info

def check_files():
    if not os.path.exists("images"):
        os.mkdir("images")
        os.mkdir("images/text2img")
        os.mkdir("images/img2img")
        
def save_image(images, seeds, info, img_type):
    check_files()

    metadata = PngInfo()
    extension = ".png"
    path = ""
    for key, value in info.items():
        metadata.add_text(str(key), str(value))

    for i in range(len(images)):
        image = images[i]
        seed = seeds[i]

        picture = isOpen(image)
        image_name = f"{seed}{extension}"

        if img_type=="t2i":
            path = os.path.join("images\\text2img", image_name)
        elif img_type=="i2i":
            path = os.path.join("images\\img2img", image_name)

        picture.save(path, pnginfo=metadata)