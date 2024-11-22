# Image Tile
## Description
Inspired by the art Kiss of Freedom, a wall covered with ceramic tiles (lippen_barcelona.jpg) in Barcelona, I made a program to do the same with own photos.

## Usage
`pip install -r requirements.txt`

Change parameters in main.py

`python main.py`

Enjoy!

## Components
On one side, **target_image** reduce the resolution of a target image *tree1.jpg*. Each pixel of the reduced image will later be replaced by an image. The reduced image **img_reduced** is then converted to a 1d list **lst_rgb_target**, where each element represent the RGB value of the pixel. The reduced image *target_scale.jpg* could be previewed by **scale_up**, which duplicate each pixel to scale up to original size.

On the other side, **image_pool** select a directory and loop through all images. Image files are cropped to a square and reduce the dimension to dim. Then it creates a dictionary **dct_rgb**, key is the relative path of the image, value is the average RGB of the image. This dictionary is split to a list of path and a list of RGB pool.

After the preparation, a cost matrix based on the distance between two RGB values between the target RGB list and the pool RGB list is created. Then a RGB list with the lowest cost is selected from pool RGB list. Then the selected RGB list is mapped to a selected path list **lst_path_sorted** and to an image list **lst_img**. Here the image is first cropped to a square and reduced resolution. Finally a big image is created by concatenate all the images in image list to the proper position.