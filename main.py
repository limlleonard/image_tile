import os
from model import target_image, image_pool, concatenate_img
from PIL import Image

# setting parameters
len_tile=80 # size of a tile
dim_max=40 # maximum dimension of the reduced target image

path_target='/home/ll/Downloads/tree1.jpg' # path of the target image
dir_pool='/media/ll/small/foto/handy' # directory of pool images
path_dict='dct_rgb_handy.json' # path to save dictionary of path:rgb

# initialize image pool
pool1=image_pool(dir_pool)
# once it is initialized, you could use read_dict instead of init_dict to save time
pool1.init_dict(len_tile, path_dict)
pool1.read_dict(path_dict)
# print(pool1.lst_failed)

# initialize target image
target1=target_image(path_target, dim_max)
img_scaled=target1.scale_up()
# img_scaled.save('target_scale.jpg')

big_image=concatenate_img(target1, pool1, dir_pool, len_tile)
if big_image is not None:
    big_image.save("target_concatenated.png")  # Save the image