import os
import glob
import numpy as np
import skimage.io as io
from skimage.color import rgb2gray
from matplotlib import pyplot as plt
from sklearn.metrics import jaccard_score

# Cargando las imagenes en un dataframe
rect_df = glob.glob(os.path.join('data/', 'rect_*.jpg'))
circ_df = glob.glob(os.path.join('data/', 'circ_*.jpg'))
validation_df = glob.glob(os.path.join('val/', '*.jpg'))
training_df = glob.glob(os.path.join('train/', '*.jpg'))

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
validation = preprocessing(validation_df) # array for validation images
training = preprocessing(training_df)

ground_truth_train = [1, 0, 0, 1, 1, 1, 0, 1, 0, 1, 0, 0] # where 1=circle, 0=rectangle
ground_truth_val = [1, 1, 0, 1, 0, 0] 


base = np.zeros((20, 20)) # creating blank canvas 


# utilizar indice de jaccard para ver precision y cobertura del algoritmo