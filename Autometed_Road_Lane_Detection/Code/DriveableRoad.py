# importing essential modules
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

video = cv.VideoCapture(r"C:\Users\Admin\Desktop\Autometed_Road_Lane_Detection/Videos/roadvideo1.mp4")

while (video.isOpened()):

    # frame by frame of video
    ret, image = video.read()

    # creating empty image of same size
    height, width, no_use = image.shape
    empty_image = np.zeros((height, width), np.uint8)

    # Applied K-Means Clustering
    Z = image.reshape((-1, 3))

    # covert to np.float32
    Z = np.float32(Z)

    # define criteria, number of clusters(K) and apply k-means()
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    K = 6
    ret, label, center = cv.kmeans(
        Z, K, None, criteria, 15, cv.KMEANS_RANDOM_CENTERS)

    # now convert back into uint8 and make original image
    center = np.uint8(center)
    res = center[label.flatten()]
    res2 = res.reshape((image.shape))

    # converted to aLUV image and made empty image, a mask
    blur = cv.GaussianBlur(res2, (15, 15), 0)
    gray = cv.cvtColor(blur, cv.COLOR_RGB2GRAY)
    LUV = cv.cvtColor(blur, cv.COLOR_RGB2LUV)
    l = LUV[:, :, 0]
    value1 = l > 80
    value2 = l < 150
    value_final = value1 & value2
    empty_image[value_final] = 255
    empty_image[LUV[:, :100, :]] = 0

    # applied bitwise-and on grayscale image and empty image to obtain road and some-other images too
    final = cv.bitwise_and(gray, empty_image)
    contours, hierchary = cv.findContours(
        final, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    final = cv.drawContours(final, contours, -1, 0, 3)

    # FURTHER MASKED THE FINAL IMAGE TO OBTAIN ONLY THE ROAD PARTICLES
    final_masked = np.zeros((height, width), np.uint8)
    value1 = final >= 91
    value2 = final <= 130
    final_masked[value1 & value2] = 255

    # APPLIED EROSION,CONTOURS AND TOP-HAT TO REDUCE NOISE
    kernel = np.ones((3, 3), np.uint8)
    final_eroded = cv.erode(final_masked, kernel, iterations=1)
    contours, hierchary = cv.findContours(
        final_eroded, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    final_masked = cv.drawContours(final_eroded, contours, -1, 0, 3)
    final_waste = cv.morphologyEx(
        final_masked, cv.MORPH_TOPHAT, kernel, iterations=2)
    final_waste = cv.bitwise_not(final_waste)
    final_masked = cv.bitwise_and(final_waste, final_masked)

    # MADE A LINE ON THE LEFT-BOTTOM OF THE PAGE
    final_masked = cv.line(final_masked, (40, height), (400, height), 255, 100)

    # USED FLOOD-FILL TO FILL IN THE SMALL BLACK LANES
    final_flood = final_masked.copy()
    h, w = final_masked.shape[:2]
    mask = np.zeros((h+2, w+2), np.uint8)
    cv.floodFill(final_flood, mask, (0, 0), 255)
    final_flood = cv.bitwise_not(final_flood)
    final_filled = cv.bitwise_or(final_masked, final_flood)

    cv.namedWindow('original', cv.WINDOW_NORMAL)
    cv.imshow('original', image)
    cv.namedWindow('tried_extraction', cv.WINDOW_NORMAL)
    cv.imshow('tried_extraction', final_filled)

    if cv.waitKey(1) & 0xFF == ord('s'):
        break

video.release()
cv.destroyAllWindows()
