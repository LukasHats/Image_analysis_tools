import numpy as np
import pandas as pd
import os
import tifffile as tiff
from skimage.filters import threshold_otsu, gaussian
from skimage.morphology import binary_dilation, disk, binary_closing, binary_opening
from scipy.ndimage import distance_transform_edt
import argparse


parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input", type=str, help="Path to the input image directory")
parser.add_argument("-q", "--quantification", type=str, help="Path to the directory where the quantification table is stored, this will get overwritten with the extended csv table. The quantificaiton table needs the x/y coords, for Steinbock that would be the regionprops.csv")
parser.add_argument("-s", "--sigma", type=int, default=5, help="Sigma for Gaussian blur on collagen channel, default is 5")
parser.add_argument("-o", "--opening", type=int, default=1, help="Opening disk option for the otsu generated bone mask, default is 1")
parser.add_argument("-c", "--closing", type=int, default=10, help="Closing disk option for the opened bone mask, default is 10")
parser.add_argument("-d", "--dilating", type=int, default=15, help="Dilating disk option for the opened and closed bone mask, default is 15")
parser.add_argument("-m", "--masks", type=str, default=None, help="Path to the output directory where the mask-multichannel TIFFs will be stored, if not provided a new output directory will be created")
args = parser.parse_args()

image_dir = args.input
quantification_dir = args.quantification

if args.masks:
    output_dir = args.masks
else:
    output_dir = os.path.join(os.path.dirname(image_dir), "bone_masks")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

# Creating a function that gets x/y coordinates from single cell table and tests, if they fall inside a created mask
def is_inside_mask(x, y, mask):
    x_int = int(round(x))
    y_int = int(round(y))
    if y_int >= mask.shape[0] or x_int >= mask.shape[1]:
        return False
    return mask[y_int, x_int]

def get_distance(row):
    x, y = round(row['centroid-1']), round(row['centroid-0'])
    y = max(0, min(y, distance_from_bone.shape[0] - 1))
    x = max(0, min(x, distance_from_bone.shape[1] - 1))
    return distance_from_bone[y, x]
for element in os.listdir(image_dir):
    try:
        image = tiff.imread(os.path.join(image_dir, element))
        collagen_channel = image[31,:,:]

        # Apply smoothing, otsu thresholding, opening/closing and dilating to get a dilated bone mask
        smoothed_image = gaussian(collagen_channel, sigma=args.sigma)
        thresh = threshold_otsu(smoothed_image)
        bone_mask_otsu = smoothed_image > thresh
        opened_mask = binary_opening(bone_mask_otsu, disk(args.opening))
        closed_mask = binary_closing(opened_mask, disk(args.closing)) 
        dilated_mask = binary_dilation(closed_mask, disk(args.dilating))

        # Get the distances to nearest bone using edt transform
        distance_from_bone = distance_transform_edt(1 - closed_mask)

        # Get the associated merged csv table and add a new column that checks if a cell is near to bone (inside the bone mask) and create column for distances
        base_filename = os.path.splitext(element)[0]
        csv_filename = base_filename + ".csv"
        csv_path = os.path.join(quantification_dir, csv_filename)
        data = pd.read_csv(csv_path)
        data['near_bone'] = data.apply(lambda row: is_inside_mask(row['centroid-1'], row['centroid-0'], dilated_mask), axis=1)

        data['distance_to_bone'] = data.apply(get_distance, axis=1)

        multichannel_image = np.stack([collagen_channel, bone_mask_otsu.astype(np.uint8), opened_mask.astype(np.uint8), closed_mask.astype(np.uint8), dilated_mask.astype(np.uint8), distance_from_bone.astype(np.uint16)], axis=0)
        output_filename = os.path.join(output_dir, base_filename + "_bone_masks.tif")
        tiff.imwrite(output_filename, multichannel_image, photometric='minisblack')
        #Overwrite csv files in quantification directory
        data.to_csv(csv_path, index=False)
    except Exception as e:
        print(f"Error processing file {element}: {e}")