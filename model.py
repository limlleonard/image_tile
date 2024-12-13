import os
import json
import math
from PIL import Image
import numpy as np
from typing import Tuple
from scipy.optimize import linear_sum_assignment

def img2sqr(path1: str, dim=80):
    '''Crop an image to a square shape and reduce the resolution to dim'''
    img = Image.open(path1)
    width, height = img.size
    square_size = min(width, height)

    left = (width - square_size) // 2
    top = (height - square_size) // 2
    right = left + square_size
    bottom = top + square_size

    img_crop = img.crop((left, top, right, bottom))
    img_sqr = img_crop.resize((dim, dim), Image.Resampling.LANCZOS)
    return img_sqr

class target_image():
    def __init__(self, path1: str, dim_max=80) -> None:
        '''Reduce the image resolution, so the longest side is smaller than dim_max.'''
        '''Return a list of (rgb) in one dimention, together with width and height to reconstruct the image
        RGB matrix are split into lines and concatenated to a 1d list'''
        self.img=Image.open(path1)
        self.width_origin, self.height_origin = self.img.size
        if self.width_origin>self.height_origin:
            scale_factor=dim_max/self.width_origin 
        else:
            scale_factor=dim_max/self.height_origin
        self.width_reduced = round(self.width_origin * scale_factor)
        self.height_reduced = round(self.height_origin * scale_factor)
        img_reduced = self.img.resize((self.width_reduced, self.height_reduced), Image.Resampling.LANCZOS)
        self.img_reduced = img_reduced.convert("RGB")
        self.lst_rgb_target=[]
        for y in range(self.height_reduced):
            for x in range(self.width_reduced):
                self.lst_rgb_target.append(self.img_reduced.getpixel((x,y)))
    
    def scale_up(self):
        '''Return the image with reduced resolution but increase the size of single pixel with scale_factor'''
        img_scaled = self.img_reduced.resize((self.width_origin, self.height_origin), Image.Resampling.NEAREST)
        return img_scaled
        # img_scaled.save("pixelated_image.jpg")

class image_pool():
    def __init__(self, dir1) -> None:
        '''Image pool, from where the tiles are selected from
        dir1: root directory to select images from'''
        self.img_extension={'.jpg','.png','jpeg','gif'}
        self.dir1=dir1
        self.dct_rgb={}
        self.lst_path, self.lst_rgb_pool=[],[]
        self.lst_failed=[]

    def init_dict(self, dim=100, path_dict='dct_rgb.json', save_json=True):
        '''
        Loop through a directory, search for image files, image files are cropped to a square and reduce the dimension to dim.
        Create a dictionary, key is the relative path of the image, value is the average RGB of the image
        dim: Dimension of the tile. path_dict: path to save the dictionary
        '''
        dct_rgb={}
        for root, _, files in os.walk(self.dir1):
            for file in files:
                if os.path.splitext(file)[1].lower() in self.img_extension:
                    path1 = os.path.join(root, file)
                    try:
                        rgb1=self.img2rgb(img2sqr(path1, dim))
                        relative_path = os.path.relpath(path1, self.dir1)
                        dct_rgb[relative_path] = rgb1
                    except:
                        self.lst_failed.append(file)

        self.dct_rgb=dct_rgb
        self.lst_path, self.lst_rgb_pool=list(dct_rgb.keys()), list(dct_rgb.values())
        if save_json:
            with open(path_dict, "w") as json_file:
                json.dump(dct_rgb, json_file)

    def read_dict(self, path_dict='dct_rgb.json'):
        '''Read saved dictionary from init_dict'''
        with open(path_dict, "r") as json_file:
            self.dct_rgb = json.load(json_file)
        self.lst_path, self.lst_rgb_pool=list(self.dct_rgb.keys()), list(self.dct_rgb.values())

    def img2rgb(self, img) -> tuple:
        '''Turn a square image into a single pixel'''
        img_array = np.array(img)
        # For images with an alpha channel (RGBA), take only the first 3 channels
        if img_array.shape[-1] == 4:
            img_array = img_array[:, :, :3]
        avg_rgb = tuple(np.mean(img_array, axis=(0, 1)).astype(int))
        return tuple(int(rgb) for rgb in avg_rgb)

def sim_color(lst_start, lst_target):
    '''Create a cost matrix where cost[i][j] is the squared distance between lst_start[i] and lst_target[j]'''
    cost_matrix = np.zeros((len(lst_start), len(lst_target)))
    for i, start in enumerate(lst_start):
        for j, end in enumerate(lst_target):
            cost_matrix[i][j] = sum((s - e) ** 2 for s, e in zip(start, end))  # Squared distance in 3D
    ind_row, ind_col = linear_sum_assignment(cost_matrix)
    # matches = [(lst_start[i], lst_target[j]) for i, j in zip(ind_row, ind_col)]
    return ind_row, ind_col

def concatenate_img(target1, pool1, dir1, len_tile=100):
    '''Calculate the cost matrix, calculate sorted list of path of image to reconstruct the target image'''
    if len(target1.lst_rgb_target)>len(pool1.lst_rgb_pool):
        return None
    ind_row, ind_col=sim_color(target1.lst_rgb_target, pool1.lst_rgb_pool) # ind_row is just [0,1,2,..., len]
    # lst_rgb_sorted=[pool1.lst_rgb_pool[a] for a in ind_col]
    lst_path_sorted=[os.path.join(dir1, pool1.lst_path[a]) for a in ind_col]
    lst_img=[img2sqr(p, len_tile) for p in lst_path_sorted]

    height_final = target1.height_reduced * len_tile
    width_final = target1.width_reduced * len_tile
    big_image = Image.new('RGB', (width_final, height_final))

    # Paste images onto the canvas
    for idx, img in enumerate(lst_img):
        row, col = divmod(idx, target1.width_reduced)  # Get row and column indices
        x_offset = col * len_tile
        y_offset = row * len_tile
        big_image.paste(img, (x_offset, y_offset))
    return big_image