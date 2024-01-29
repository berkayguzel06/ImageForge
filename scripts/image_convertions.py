import os
from PIL import Image
import matplotlib.pyplot as plt

def save_image(images, seeds, info, img_type):
    extension = ".jpg"
    all_info = ""
    path = ""
    for key, value in info.items():
        all_info += ("-"+str(value))

    if not os.path.exists("images"):
        os.mkdir("images")
        os.mkdir("images/text2img")
        os.mkdir("images/img2img")

    for i in range(len(images)):
        image = images[i]
        seed = seeds[i]

        if isinstance(image, Image.Image):
            picture = image
        else:
            picture = Image.open(image)

        image_name = f"{seed}{all_info}{extension}"

        if img_type=="t2i":
            path = os.path.join("images\\text2img", image_name)
        elif img_type=="i2i":
            path = os.path.join("images\\img2img", image_name)

        picture.save(path)

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