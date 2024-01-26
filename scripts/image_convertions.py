import os
from PIL import Image
import matplotlib.pyplot as plt

def save_image(images, seeds, info):
    extension = ".jpg"
    all_info = ""
    
    for key, value in info.items():
        all_info += ("-"+str(value))

    if not os.path.exists("images"):
        os.mkdir("images")
        os.mkdir("images/plot")
        os.mkdir("images/image")

    for i in range(len(images)):
        image = images[i]
        seed = seeds[i]

        if isinstance(image, Image.Image):
            picture = image
        else:
            picture = Image.open(image)

        image_name = f"{seed}{all_info}{extension}"
        path = os.path.join("images/image", image_name)

        picture.save(path)