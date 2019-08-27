#!/usr/bin/env python3


# This program will take a directory of images and remove any objects that are moving
# The program can take the argument --no-sky to skip th sky masking stage
# this is usefull when taking pictures indoors, or when keeping clouds is not important
#
# This was created by Brenton Fairless, Grant Douglas and Marshall Asch
#

import cv2
import os
import numpy as np
import math
import scipy.ndimage as nd
import pylab
import sys
import uuid
import shutil
from scipy import spatial




baseDirectory=""

#
# This function will compute the median filter of the images
# file1 is considered to be the "good" position that it will attempt to align
# the rest of the images with.
#
# if showSky is set to 0 then it will not attempt to identify and mask the sky region
# of the image
#
def medianFilter(fileName1, files, showSky=1):
    imgOriginal = cv2.imread(fileName1)

    images = [cv2.imread(name) for name in files]

    # sky removal
    if showSky == 1:

        if os.path.exists(os.path.join(baseDirectory, "skyIntermediate")) :
            shutil.rmtree(os.path.join(baseDirectory, 'skyIntermediate'))

        os.mkdir(os.path.join(baseDirectory, "skyIntermediate"))

        images = [skyRecognition(img) for img in images]

        print("done sky recognition")

        imgOne = skyRecognition(imgOriginal)

        # image registration
        images = [imgOne] + [alignImages(img, imgOne) for img in images]

        print("done aligning")
    else:
        images = [imgOriginal] + [alignImages(img, imgOriginal) for img in images]

    # perform the actual median filter
    together = np.array(images)
    final = np.median(together, axis=0)

    img = final[:]

    if showSky == 1:
        img[np.where(images[0] == 0)] = imgOriginal[np.where(images[0] == 0)]


    # for k in img:
    #     for j in k:
    #         if j == [255, 255, 255]:


    return img


##
# This will identify the sky region of the image and will return the image with the
# sky portion masked, it will also save some intermediate files
#
def skyRecognition(image, thresh_max=900, thresh_min=5, step=5):


    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    gradient = np.hypot(cv2.Sobel(gray, cv2.CV_64F, 1, 0), cv2.Sobel(gray, cv2.CV_64F, 0, 1))

    cv2.imwrite(os.path.join(baseDirectory, "intermediate", "sobel_merge.jpg"), gradient)

    Jnmax = 0
    bopt = None

    n = ((thresh_max - thresh_min) // step) + 1

    for val in range(1, 120):
        t = thresh_min + ((thresh_max - thresh_min) // n - 1) * (val - 1)
        temp = calc_border(gradient, t)

        jn = energy(temp, image)
        if jn > Jnmax:
            Jnmax = jn
            bopt = temp

    cv2.imwrite(os.path.join(baseDirectory, "intermediate", "final.png"), bopt)

    return display_mask(bopt, image)


# This will mask out the sky region and return the masked image, it will also
# save it to the skyIntermediate directory
#
def display_mask(b, image, color=[0, 0, 255]):
    result = image.copy()
    overlay = np.full(image.shape, color, image.dtype)

    final = cv2.addWeighted(
        cv2.bitwise_and(overlay, overlay, mask=make_mask(b, image)),
        1,
        image,
        1,
        0,
        result
    )

    final[final[:, :, 2] < 255] = 0

    for k in range(0, final.shape[1]):
        for j in range(0, final.shape[0]):
            if final[j, k, 2] == 255:
                final[j, k, 2] = image[j, k, 2]

    cv2.imwrite(os.path.join(baseDirectory, "skyIntermediate",  str(uuid.uuid4()) + ".png"), final)

    return final


def make_mask(b, image):
    mask = np.zeros((image.shape[0], image.shape[1], 1), dtype=np.uint8)
    for xx, yy in enumerate(b):
        mask[yy:, xx] = 255

    return mask

def energy(b_tmp, image):
    sky_mask = make_mask(b_tmp, image)

    ground = np.ma.array(image, mask=cv2.cvtColor(cv2.bitwise_not(sky_mask), cv2.COLOR_GRAY2BGR)).compressed()
    sky = np.ma.array(image, mask=cv2.cvtColor(sky_mask, cv2.COLOR_GRAY2BGR)).compressed()

    ground.shape = (ground.size//3, 3)
    sky.shape = (sky.size//3, 3)

    sigma_g, mu_g = cv2.calcCovarMatrix(
        ground,
        None,
        cv2.COVAR_NORMAL | cv2.COVAR_ROWS | cv2.COVAR_SCALE
    )

    sigma_s, mu_s = cv2.calcCovarMatrix(
        sky,
        None,
        cv2.COVAR_NORMAL | cv2.COVAR_ROWS | cv2.COVAR_SCALE
    )

    y = 2

    return 1 / (
        (y * np.linalg.det(sigma_s) + np.linalg.det(sigma_g)) +
        (y * np.linalg.det(np.linalg.eig(sigma_s)[1]) +
            np.linalg.det(np.linalg.eig(sigma_g)[1]))
    )


def calc_border(image, t):
    b = np.full(image.shape[1], image.shape[0])

    for x in range(image.shape[1]):
        pos = np.argmax(image[:, x] > t)

        if pos > 0:
            b[x] = pos

    return b


# take an image and alight it to the reference image using a Affine transform
#
def alignImages(im2, reference):

    # Convert images to grayscale
    im1_gray = cv2.cvtColor(reference,cv2.COLOR_BGR2GRAY)
    im2_gray = cv2.cvtColor(im2,cv2.COLOR_BGR2GRAY)

    # Find size of the base image
    sz = reference.shape

    # Define the motion model, this one actually finished, MOTION_AFFINE did not
    # warp_mode = cv2.MOTION_AFFINE
    # warp_matrix = np.eye(2, 3, dtype=np.float32)

    warp_mode = cv2.MOTION_AFFINE
    warp_matrix = np.eye(2, 3, dtype=np.float32)

    # Specify the number of iterations.
    number_of_iterations = 500

    # increment threshhold
    # in the correlation coefficient between two iterations
    # a smaller number here means more accurate
    termination_eps = 1e-4;

    # use the count or the accuracy to terminate
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, number_of_iterations,  termination_eps)

    # Run the ECC algorithm. The results are stored in warp_matrix.
    (cc, warp_matrix) = cv2.findTransformECC (im1_gray,im2_gray,warp_matrix, warp_mode, criteria)

    aligned = cv2.warpAffine(im2, warp_matrix, (sz[1],sz[0]), flags=cv2.INTER_CUBIC | cv2.WARP_INVERSE_MAP)

    return aligned





if __name__ == "__main__":


    baseDirectory = os.path.dirname(os.path.abspath(__file__))

    showSky = 1
    dirName = ""

    if len(sys.argv) == 3 and sys.argv[1] == "--no-sky":
        showSky = 0
        dirName = sys.argv[2]
        print("no Sky")
    elif len(sys.argv) == 2:
        dirName = sys.argv[1]
    else:
        print("error bad args")
        exit(-1)


    imageList = [dirName + "/" + f for f in os.listdir(dirName)]

    first = imageList[0];
    del imageList[0]

    result = medianFilter(first, imageList, showSky)


    if showSky == 1:
        cv2.imwrite(os.path.join(baseDirectory, "final_skyRemoved.png"), result)
    else:
        cv2.imwrite(os.path.join(baseDirectory, "final_noSky.png"), result)



