import os
from PIL import Image
import numpy as np

dir_path = os.path.join(os.getcwd(), 'SR_dataset', 'Set5')
set_dir_path = os.path.join(os.getcwd(), 'SR_dataset', 'val_Set5')
image_file_list = os.listdir(dir_path)
image_list = []
for image_name in image_file_list:
    file_path = os.path.join(dir_path, image_name)
    im = Image.open(file_path)
    im_size = im.size
    print(im_size)
    im = im.resize((int(im_size[0]/2), int(im_size[1]/2)))
    im = im.resize(im_size)
    output_path = os.path.join(set_dir_path, image_name)
    im.save(output_path)
