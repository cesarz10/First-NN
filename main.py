from cgitb import grey
import os
import glob
import numpy as np
import skimage.io as io
from skimage.color import rgb2gray
from matplotlib import pyplot as plt
from skimage.filters import threshold_otsu


# Cargando las imagenes en un dataframe
rect_df = glob.glob(os.path.join('data/', 'rect_*.jpg'))
circ_df = glob.glob(os.path.join('data/', 'circ_*.jpg'))

# Function for the preprocessing of the images
def preprocessing(df):
    arr = []

    for image in df:
        gray = rgb2gray(io.imread(image)) # converting images to grayscale
        bina = gray > 0.5 # binarizing image color
        arr.append(bina)

    return arr

rects = preprocessing(rect_df) # array for rectangle images
circles = preprocessing(circ_df) # array for circle images

plt.imshow(circles[3], cmap='gray')
plt.show()