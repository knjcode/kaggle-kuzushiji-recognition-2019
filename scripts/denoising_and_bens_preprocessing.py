import os
import sys
import numpy as np # linear algebra
import cv2 # image processing

from glob import glob
from multiprocessing import Pool

# Slightly modified copy of https://www.kaggle.com/hanmingliu/denoising-ben-s-preprocessing-better-clarity

# making sure result is reproducible
SEED = 2019
np.random.seed(SEED)

def read_image(image):
    '''
        Simply read a single image and convert it RGB in opencv given its filename.
    '''

    return cv2.cvtColor(cv2.imread(image), cv2.COLOR_BGR2RGB)


def apply_ben_preprocessing(image):
    '''
        Apply Ben's preprocessing on a single image in opencv format
    '''

    return cv2.addWeighted(image, 4, cv2.GaussianBlur(image, (0,0), 10), -4, 128)


def apply_denoising(image):
    '''
        Apply denoising on a single image given it in opencv format.
        Denoising is done using twice the recommended strength from opencv docs.
    '''

    return cv2.fastNlMeansDenoisingColored(image, None, 20, 20, 7, 21)


target = sys.argv[1]
if not target in ['train', 'test']:
  print("specify 'train' or 'test'")
  sys.exit(1)

output_dir = f'./input/denoised_{target}/'
target_files = glob(f"./input/{target}_images/*.jpg")

os.makedirs(output_dir, exist_ok=True)

def denoise_and_bens_prepro(filepath):
    filename = filepath.split('/')[-1]
    filename = filename.replace('.jpg', '.png')
    img = read_image(filepath)
    prepro = apply_ben_preprocessing(img)
    after = apply_denoising(prepro)
    save_path = output_dir + filename
    cv2.imwrite(save_path, after)
    print(save_path)


process_num = 32
p = Pool(process_num)
p.map(denoise_and_bens_prepro, target_files)
p.close()
