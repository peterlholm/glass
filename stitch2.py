"glass calculator"
from pathlib import Path
import math
from matplotlib import pyplot as plt
import cv2 as cv
import numpy as np
#from collections import OrderedDict

# pylint: disable=invalid-name

FEATURES = "SIFT"
MATCHER = "BF"    # FLANN

def plot_image(img, figsize_in_inches=(5,5)):
    "plot single image"
    fig, ax = plt.subplots(figsize=figsize_in_inches)   # pylint: disable=unused-variable, invalid-name
    ax.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))
    plt.show()

def plot_images(imgs, figsize_in_inches=(5,5)):
    "plot more images"
    fig, axs = plt.subplots(1, len(imgs), figsize=figsize_in_inches) # pylint: disable=unused-variable, invalid-name
    for col, img in enumerate(imgs):
        axs[col].imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))
    plt.show()

def process(img1, img2):
    "process the images and return the strength of the glass"
    # read image
    full_img1 = cv.imread(str(img1))
    full_img2 = cv.imread(str(img2))
    work_scale = 0.8
    img1 = cv.resize(src=full_img1, dsize=None, fx=work_scale, fy=work_scale, interpolation=cv.INTER_LINEAR_EXACT)
    img2 = cv.resize(src=full_img2, dsize=None, fx=work_scale, fy=work_scale, interpolation=cv.INTER_LINEAR_EXACT)
    gray1 = cv.cvtColor(full_img1, cv.COLOR_BGR2GRAY)
    gray2 = cv.cvtColor(full_img2, cv.COLOR_BGR2GRAY)

    if FEATURES=="ORB":
        featurefinder = cv.ORB_create()
        kp1, des1 = featurefinder.detectAndCompute(gray1, None)
        kp2, des2 = featurefinder.detectAndCompute(gray2, None)
        #print("KP", dir(kp1[0]))
        #print("KP", keypoints1[0])
        #print("DESC", dir(des1[0]))
        flags = cv.DrawMatchesFlags_DEFAULT
        #flags = cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
        img1k = cv.drawKeypoints(img1, kp1, None, flags=flags)
        cv.imshow("ORG", img1k)
        cv.waitKey(0)

    # sift
    if FEATURES=="SIFT":
        sift = cv.SIFT_create()
        #sift = cv.xfeatures2d.SIFT_create()
        kp1, des1 = sift.detectAndCompute(gray1,None)
        kp2, des2 = sift.detectAndCompute(gray2,None)
        #flags = cv.DrawMatchesFlags_DEFAULT
        flags = cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
        #flags = cv.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS  #, cv.DRAW_MATCHES_FLAGS_DRAW_OVER_OUTIMG, cv.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS
        img1s = cv.drawKeypoints(img1, kp1, None, flags=flags)
        cv.imshow('SIFT', img1s)
        #cv.waitKey(0)

    if MATCHER=="FLANN":
        if FEATURES=="SIFT":
            FLANN_INDEX_KDTREE = 1
            index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
            search_params = dict(checks=50)   # or pass empty dictionary
            flann = cv.FlannBasedMatcher(index_params,search_params)
            matches = flann.knnMatch(des1, des2, k=2)
            # Need to draw only good matches, so create a mask
            matchesMask = [[0,0] for i in range(len(matches))]
            # ratio test as per Lowe's paper
            for i,(m,n) in enumerate(matches):
                if m.distance < 0.7*n.distance:
                    matchesMask[i]=[1,0]
            draw_params = dict(matchColor = (0,255,0),
                            singlePointColor = (255,0,0),
                            matchesMask = matchesMask,
                            flags = cv.DrawMatchesFlags_DEFAULT)
            img3 = cv.drawMatchesKnn(img1,kp1,img2,kp2,matches,None,**draw_params)
            cv.imshow('FLANN', img3)
            cv.waitKey(0)


        # FLANN_INDEX_KDTREE = 0
        # index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
        # search_params = dict(checks = 50)
        # match = cv.FlannBasedMatcher(index_params, search_params)
        # matches = match.knnMatch(des1,des2,k=2)

    if MATCHER=="BF":
        if FEATURES=="ORB":
            bf_matcher = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
            matches = bf_matcher.match(des1, des2)
            #matches = bf_matcher.knnMatch(des1, des2, 2)
            matches = sorted(matches, key=lambda x:x.distance)
            imgm = cv.drawMatches(img1, kp1, img2, kp2, matches[:10], None, flags=2)
            cv.imshow('MATCH', imgm)
            cv.waitKey(0)

        if FEATURES=="SIFT":
            bf_matcher = cv.BFMatcher()
            matches = bf_matcher.knnMatch(des1, des2, k=2)
            #matches = sorted(matches, key=lambda x:x.distance)
            good = []
            for m,n in matches:
                if m.distance < 0.5*n.distance:
                    good.append(m)
            #print("Good", good)
            draw_params = dict(matchColor = (0,255,0), # draw matches in green color
                   singlePointColor = None,
                   flags = 2)
            #imgm = cv.drawMatchesKnn(img1, kp1, img2, kp2, good, None, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
            imgm = cv.drawMatches(img1, kp1, img2, kp2, good, None, **draw_params)
            cv.imshow('SIFT MATCH', imgm)
            cv.waitKey(0)



        # HOMOGRAFI
        #print(good)
        MIN_MATCH_COUNT = 10
        if len(good)>MIN_MATCH_COUNT:
            src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
            dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)

            M, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC,5.0)


            if np.shape(M) == ():
                print( "No transformation possible" )
            else:
                ## derive rotation angle from homography
                theta = - math.atan2(M[0,1], M[0,0]) * 180 / math.pi
                print("M", M)
                print("Vinkel", theta)
                #print("Mask", mask)

            h,w = gray1.shape
            pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
            dst = cv.perspectiveTransform(pts,M)

            gray2 = cv.polylines(gray2,[np.int32(dst)],True,255,3, cv.LINE_AA)
            cv.imshow("original_image_overlapping.jpg", gray2)
            cv.waitKey(0)
        else:
            print ("Not enough matches are found - %d/%d" % (len(good),MIN_MATCH_COUNT))










        # for m in matches[:5]:
        #     print(m)
        #     print(m.distance)
        #     print(m.trainIdx)
        #     print(m.queryIdx)
        #     print(m.imgIdx)

    return
    # draw_params = dict(matchColor = (0,255,0), # draw matches in green color
    #                singlePointColor = None,
    #                flags = 2)

    # match_img = cv.drawMatches(img1, kp1, img2, kp2, matches[:20], None, flags=2)
    # cv.imshow("match", match_img)
    # cv.waitKey(0)
    # #plot_images([img1,img2,img4, match_img])
    # return


    # #matcher = cv.detail_AffineBestOf2NearestMatcher
    # #   (*features_matcher_)(features_, pairwise_matches_, matching_mask_);

    # matcher = cv.detail_BestOf2NearestMatcher(False, 0.3, 6, 6, 3.0)

    # print(matcher, type(matcher))
    # matches = matcher(des1,des2)
    # print(matches)


    # return
    #if matcher_type == "affine":
    # matcher = cv.detail_AffineBestOf2NearestMatcher(False, try_cuda, match_conf)
    # matches = matcher.match(descriptor1, descriptor2)
    # matches = sorted(matches, key = lambda x:x.distance)
    # img3 = cv.drawMatches(gray1, keypoints1, gray2, keypoints2, matches[500:600], gray2, flags=2)

    # matcher = cv.detail_BestOf2NearestMatcher(try_cuda, match_conf)
    # matches = matcher.match(descriptor1, descriptor2)
    # matches = sorted(matches, key = lambda x:x.distance)
    # img3 = cv.drawMatches(gray1, keypoints1, gray2, keypoints2, matches[500:600], gray2, flags=2)
    # cv.imshow('SIFT', img3)
    # cv.waitKey()

    # matcher = cv.detail_BestOf2NearestRangeMatcher(range_width, try_cuda, match_conf)
    # matches = matcher.match(descriptor1, descriptor2)
    # matches = sorted(matches, key = lambda x:x.distance)
    # img3 = cv.drawMatches(gray1, keypoints1, gray2, keypoints2, matches[500:600], gray2, flags=2)
    # cv.imshow('SIFT', img3)
    # cv.waitKey()

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
    # img_feat = cv.detail.computeImageFeatures2(cv.ORB_create(), img)
    # print("Features", img_feat)

if __name__ == '__main__':
    folder = Path(__file__).parent /"testdata/test1_small"
    # pic1 = folder / "DSC_0014.JPG"
    # pic2 = folder / "DSC_0015.JPG"
    pic1 = folder / "boat1.jpg"
    pic2 = folder / "boat2.jpg"
    print(folder)
    process(pic1,pic2)
    cv.destroyAllWindows()
