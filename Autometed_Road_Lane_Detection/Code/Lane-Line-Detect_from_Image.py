# importing essential modules
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

image = cv.imread(r"C:\Users\Admin\Desktop\Autometed_Road_Lane_Detection\Images/LaneRoad10.png")


def region_of_interest(img, vertices):
    mask = np.zeros_like(img)

    # create a match color with the same color channel count
    match_mask_color = 255

    # fill inside the polygon using the fillpoly method
    cv.fillPoly(mask, vertices, match_mask_color)

    mask_image = cv.bitwise_and(img, mask)
    return mask_image

def draw_lines(img, lines):
    img = np.copy(img)
    # blank image
    blank_image = np.zeros((img.shape[0], img.shape[1], 3), dtype= np.uint8)

    for line in lines:
        for x1,y1,x2,y2 in line:
            cv.line(blank_image, (x1,y1), (x2,y2), (0, 255, 0), thickness=5)

    # merge images
    img = cv.addWeighted(img, 0.8, blank_image, 1, 0.0)
    return img

# convert image into RGB formate
image = cv.cvtColor(image, cv.COLOR_BGR2RGB)

# detect image height and width indivisually
image_height = image.shape[0]
image_width = image.shape[1]

# define region of interest
region_of_interest_vertices = [
    (0, image_height), (image_width/2, image_height/2), (image_width, image_height)]


# convert image to gray scale
gray_image = cv.cvtColor(image, cv.COLOR_RGB2GRAY)

# edge detection
canny_image = cv.Canny(gray_image, 100, 200)

cropped_image = region_of_interest(
    canny_image, np.array([region_of_interest_vertices], np.int32), )

# hough line transform
lines = cv.HoughLinesP(cropped_image, rho=6, theta=np.pi/60, threshold=160, lines=np.array([]), minLineLength=50, maxLineGap=25)
image_with_lines = draw_lines(image, lines)

plt.imshow(image_with_lines)
plt.show()
