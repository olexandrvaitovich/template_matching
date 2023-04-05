import numpy as np
import cv2
from glob import glob
from skimage import measure
import pandas as pd
import os
from tqdm import tqdm
import config
import argparse


def area_filter(min_area, input_image):
    """Filters out small interconnected areas of pixels and leaves only ones
       with bigger amount of pixels than min_area

    Parameters
    ----------
    min_area : int
        minimal amount of interconnected pixels
    input_image : numpy.array
        The image itself

    Returns
    -------
    numpy array
        Binary image with white background and black blueprints
    """
    # Perform an area filter on the binary blobs:
    components_number, labeled_image, component_stats, component_centroids = \
        cv2.connectedComponentsWithStats(input_image, connectivity=4)

    # Get the indices/labels of the remaining components based on the area stat
    # (skip the background component at index 0)
    remaining_component_labels = [i for i in range(1, components_number) if component_stats[i][4] >= min_area]

    # Filter the labeled pixels based on the remaining labels,
    # assign pixel intensity to 255 (uint8) for the remaining pixels
    filtered_image = np.where(np.isin(labeled_image, remaining_component_labels), 255, 0).astype('uint8')

    return filtered_image


def get_bottom_mask(img_gray, min_area1=200000, min_area2=50000, erode_iterations=2, dilate_iterations=4):
    """Creates the mask of bottom half of each blueprint

    Parameters
    ----------
    img_gray : numpy.array
        The grayscaled image itself
    min_area1 : int, optional
        A minimal number of pixels that is considered as a distinct area in step1(default is
        200000)
    min_area2 : int, optional
        A minimal number of pixels that is considered as a distinct area in step2(default is
        50000)
    erode_iterations : int, optional
        A number of erosion iterations to remove noise(default is 2)
    dilate_iterations : int, optional
        A number of dilation iterations to restore blueprint size(default is 2)
    

    Returns
    -------
    numpy.array
        A mask with marked bottoms of each blueprint
    """
    # Separating blueprints
    mask = cv2.inRange(img_gray, 255, 255)

    mask = area_filter(min_area1, mask)

    mask = cv2.erode(mask, None, iterations=erode_iterations)
    mask = cv2.dilate(mask, None, iterations=dilate_iterations)

    mask = 255 - area_filter(min_area2, 255 - mask)

    labels = measure.label(mask // 255, background=1)

    # Creating mask

    bottom_mask = np.zeros(labels.shape)

    for i in np.unique(labels):
        if i == 0:
            continue
        ys_spot, xs_spot = np.where(labels == i)

        limit = (np.max(ys_spot) + np.min(ys_spot)) / 2

        bottom_mask[(ys_spot[np.where(ys_spot > limit)], xs_spot[np.where(ys_spot > limit)])] = 1

    return bottom_mask


def run(img_path):
    """Runs all the functions. Saves result coordinates into csv file. Saves the picture
       with marked boxes.
       
    Parameters
    ----------
    img_path : str
        Path to the image

    Returns
    -------
    None
    """
    print(f'\nProcessing {os.path.abspath(img_path)}\n')
    
    os.makedirs(config.save_path, exist_ok=True)

    img_orig = cv2.imread(img_path)
    img_gray = cv2.cvtColor(img_orig.copy(), cv2.COLOR_BGR2GRAY)

    result_dict = {'start_y': [], 'end_y': [], 'start_x': [], 'end_x': []}

    bottom_mask = get_bottom_mask(img_gray)  # Masking botton of each blueprint to avoid detecting boxes there

    viz_img = img_orig.copy()

    img = cv2.cvtColor(img_orig, cv2.COLOR_BGR2GRAY)

    cover_mask = np.zeros(viz_img.shape[:2])  # Mark used areas for duplicates removal

    templates = sorted(glob(f'{config.templates_path}/*.png'), key=lambda x: 'half' in x)

    for t in tqdm(templates, desc="Applying templates"):
        template_orig = cv2.imread(t)
        w = template_orig.shape[1]
        # Width of the template not always exactly matches actual width of wall cabinets on the blueprints
        # So I try different width number
        for i in reversed(range(int(-w + w / config.size_mult), int(w - w / config.size_mult))):
            template = cv2.cvtColor(template_orig, cv2.COLOR_BGR2GRAY)
            template = cv2.resize(template, (template.shape[1] + i, template.shape[0]))

            result = cv2.matchTemplate(img, template, cv2.TM_CCOEFF_NORMED)

            # Higher confidence goes first
            ys, xs = np.where(result >= config.conf_thr)
            order = result[(ys, xs)].argsort()[::-1]
            ys, xs = ys[order], xs[order]

            for y, x in zip(ys, xs):
                # Check whether the area is already used by another sample
                # Wall cabinets are never situated at the bottom of the image so we don't need those detections
                if cover_mask[y, x] != 1 and bottom_mask[y, x] != 1:
                    start_x, start_y = x, y
                    end_x = start_x + template.shape[1]
                    end_y = start_y + template.shape[0]

                    viz_img = cv2.rectangle(viz_img, (start_x, start_y), (end_x, end_y), (255, 0, 0), 3)

                    result_dict['start_y'].append(start_y)
                    result_dict['end_y'].append(end_y)
                    result_dict['start_x'].append(start_x)
                    result_dict['end_x'].append(end_x)

                    cover_mask[start_y - 10: end_y - 10, start_x - 10: end_x - 10] = 1

    cv2.imwrite(f'{config.save_path}/{os.path.basename(img_path)}_marked_image.png', viz_img)
    df = pd.DataFrame(result_dict)
    df.to_csv(f'{config.save_path}/{os.path.basename(img_path)}_result_df.csv', index=False)
    
    print(f'\nResults are saved to {os.path.abspath(config.save_path)}\n')
    


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Detects wall cabinets in images')
    parser.add_argument('--img', required=True, help='path to the image/images dir')

    args = parser.parse_args()

    images = glob(f'{args.img}/*') if os.path.isdir(args.img) else [args.img]

    for im in images:
        try:
            run(im)
        except Exception as e:
            print(e)
            continue
