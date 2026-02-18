import os
import numpy as np
import cv2
from glob import glob
from tqdm import tqdm
import imageio

""" Create a directory """
def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def load_data(path):
    train_x = sorted(glob(os.path.join(path, "training", "images", "*.tif")))
    train_y = sorted(glob(os.path.join(path, "training", "1st_manual", "*.gif")))

    test_x = sorted(glob(os.path.join(path, "test", "images", "*.tif")))
    test_y = sorted(glob(os.path.join(path, "test", "1st_manual", "*.gif")))
    
    return (train_x, train_y), (test_x, test_y)

def process_data(images, masks, save_path):
    size = (512, 512)

    image_path = os.path.join(save_path, "image")
    mask_path = os.path.join(save_path, "mask")
    create_dir(image_path)
    create_dir(mask_path)

    for idx, (x, y) in tqdm(enumerate(zip(images, masks)), total=len(images)):
        """ Extracting the name """
        name = os.path.splitext(os.path.basename(x))[0]
        
        """ Reading image and mask """
        x = cv2.imread(x, cv2.IMREAD_COLOR)
        y = imageio.mimread(y)[0]

        # Resize images
        x = cv2.resize(x, size)
        y = cv2.resize(y, size)

        # Save images
        image_name = f"{name}.png"
        mask_name = f"{name}.png"

        image_save_path = os.path.join(save_path, "image", image_name)
        mask_save_path = os.path.join(save_path, "mask", mask_name)

        cv2.imwrite(image_save_path, x)
        cv2.imwrite(mask_save_path, y)

if __name__ == "__main__":
    """ Seeding """
    np.random.seed(42)

    """ Load the data """
    data_path = "./DRIVE/"
    (train_x, train_y), (test_x, test_y) = load_data(data_path)

    print(f"Train: {len(train_x)} - {len(train_y)}")
    print(f"Test: {len(test_x)} - {len(test_y)}")

    """ Create directories to save the processed data """
    path = 'new_data'
    create_dir(path)
    create_dir(path + '/train')
    create_dir(path + '/test')

    """ Process data """
    process_data(train_x, train_y, path + '/train')
    process_data(test_x, test_y, path + '/test')