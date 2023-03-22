"glass calculator"
import math
from pathlib import Path
from matplotlib import pyplot as plt
import cv2 as cv
import numpy as np
from collections import OrderedDict


def plot_image(img, figsize_in_inches=(5,5)):
    "plot single image"
    fig, ax = plt.subplots(figsize=figsize_in_inches)
    ax.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))
    plt.show()
    
def plot_images(imgs, figsize_in_inches=(5,5)):
    "plot more images"
    fig, axs = plt.subplots(1, len(imgs), figsize=figsize_in_inches)
    for col, img in enumerate(imgs):
        axs[col].imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))
    plt.show()


def process(img1, img2):
    "process the images and return the strength of the glass"
    full_img1 = cv.imread(str(img1))
    full_img2 = cv.imread(str(img2))

    work_scale_ = 1
    seam_work_aspect_ = 1
    seam_scale_ = 1
    is_work_scale_set = False
    s_seam_scale_set = False
    seam_est_resol_ = 0.1

    work_scale = math.sqrt(0.6 * 1e6 / full_img1.size)
    print(work_scale_)
    is_work_scale_set = True
    #features_.resize(imgs_.size());
    #seam_est_imgs_.resize(imgs_.size());
    #full_img_sizes_.resize(imgs_.size());

    # sift = cv.SIFT_create()
    # bf = cv.BFMatcher(cv.NORM_L2, crossCheck=True)

    #work_scale = 0.102
    #work_scale = 0.2

    #print(full_img1)

    print(full_img1.size)
    print(math.sqrt(0.6 * 1000000 / full_img1.size))
    print(math.sqrt(0.1* 1000000 / full_img1.size))
    img1 = cv.resize(src=full_img1, dsize=None, fx=work_scale, fy=work_scale, interpolation=cv.INTER_LINEAR_EXACT)
    img2 = cv.resize(src=full_img2, dsize=None, fx=work_scale, fy=work_scale, interpolation=cv.INTER_LINEAR_EXACT)
   
    seams_scale = min(1.0, math.sqrt(seam_est_resol_ * 1e6 / full_img1.size))
    print("seam_scale", seams_scale)
    seam_work_aspect_ = seam_scale_ / work_scale_
    print("seam_work_aspect", seam_work_aspect_)


    # gray1 = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)
    # gray2 = cv.cvtColor(img2, cv.COLOR_BGR2GRAY)

    #detail::computeImageFeatures(features_finder_, feature_find_imgs, features_, feature_find_masks);

    #keypoints = cv.Feature2D.detect()
    featurefinder = cv.ORB_create()
    keypoints1 = featurefinder.detect(img1)
    des1 = featurefinder.compute(img1, keypoints1)
    #print("keypoints", keypoints)
    #print(des1)
    img3 = cv.drawKeypoints(img1, keypoints1, 0, (0, 0, 255))

    keypoints2, des2 = featurefinder.detectAndCompute(img2, None)
    # keypoints2 = featurefinder.detect(img2)
    # des2 = featurefinder.compute(img2, keypoints2)

    #print("keypoints", keypoints)
    flags = cv.DrawMatchesFlags_DEFAULT
    #flags = cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS 
    #flags = cv.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS  #, cv.DRAW_MATCHES_FLAGS_DRAW_OVER_OUTIMG, cv.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS 
    img4 = cv.drawKeypoints(img1, keypoints2, None, (0, 0, 255), flags)

    cv.imshow('keypoints', img4)
    cv.waitKey(0)

    # brute force matching

    bf_matcher = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)

    matches = bf_matcher.match(des1, des2)
    print (matches )

    plot_images([img1,img2,img3,img4])

    #matcher = cv.detail_AffineBestOf2NearestMatcher
    #   (*features_matcher_)(features_, pairwise_matches_, matching_mask_);
 
    matcher = cv.detail_BestOf2NearestMatcher(False, 0.3, 6, 6, 3.0)

    print(matcher, type(matcher))
    matches = matcher(keypoints1,keypoints2)
    print(matches)


  
    #featurefinder.detectAndCompute(gray1,None)

    # kp = sift.detect(img1, None)
    # img3 = cv.drawKeypoints(img1, kp, 0, (0, 0, 255))
    #                              #flags=cv.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)

    keypoints1, descriptor1 = featurefinder.detectAndCompute(img1, None)
    keypoints2, descriptor2 = sift.detectAndCompute(gray2, None)

    matches = bf.match(descriptor1, descriptor2)
    matches = sorted(matches, key = lambda x:x.distance)

    img3 = cv.drawMatches(gray1, keypoints1, gray2, keypoints2, matches[500:600], gray2, flags=2)              
    cv.imshow('SIFT', img3)
    cv.waitKey()

    try_cuda = False
    match_conf = 0.3
    match_conf = 0.65
 
    #if matcher_type == "affine":
    matcher = cv.detail_AffineBestOf2NearestMatcher(False, try_cuda, match_conf)
    matches = matcher.match(descriptor1, descriptor2)
    matches = sorted(matches, key = lambda x:x.distance)
    img3 = cv.drawMatches(gray1, keypoints1, gray2, keypoints2, matches[500:600], gray2, flags=2)              
    cv.imshow('SIFT', img3)
    cv.waitKey()

    matcher = cv.detail_BestOf2NearestMatcher(try_cuda, match_conf)
    matches = matcher.match(descriptor1, descriptor2)
    matches = sorted(matches, key = lambda x:x.distance)
    img3 = cv.drawMatches(gray1, keypoints1, gray2, keypoints2, matches[500:600], gray2, flags=2)              
    cv.imshow('SIFT', img3)
    cv.waitKey()

    matcher = cv.detail_BestOf2NearestRangeMatcher(range_width, try_cuda, match_conf)
    matches = matcher.match(descriptor1, descriptor2)
    matches = sorted(matches, key = lambda x:x.distance)
    img3 = cv.drawMatches(gray1, keypoints1, gray2, keypoints2, matches[500:600], gray2, flags=2)              
    cv.imshow('SIFT', img3)
    cv.waitKey()
 
    # seam_megapix = 0.1
    # seam_scale = min(1.0, np.sqrt(seam_megapix * 1e6 / (full_img1.shape[0] * full_img1.shape[1])))
    # seam_work_aspect = seam_scale / work_scale
    # is_seam_scale_set = True
    # # FEATURES_FIND_CHOICES = OrderedDict()
    # FEATURES_FIND_CHOICES['orb'] = cv.ORB.create
    # default=list(FEATURES_FIND_CHOICES.keys())[0]
    # print(default)
    # print(type(default))
    #finder = cv.ORB.create
    img_feat = cv.detail.computeImageFeatures2(cv.ORB_create(), img)
    print("Features", img_feat)



if __name__ == '__main__':

    folder = Path(__file__).parent /"testdata/mytest"
    pic1 = folder / "DSC_0014.JPG"
    pic2 = folder / "DSC_0015.JPG"


    print(folder)
    process(pic1,pic2)
   

    cv.destroyAllWindows()
