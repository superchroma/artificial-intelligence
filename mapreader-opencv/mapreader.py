#!/usr/bin/env python3
# This is template for your program, for you to expand with all the correct
# functionality.
import numpy as np
import sys, math, cv2

#-----------------------------------------------------------------------------#
# Main program.

# Ensure we were invoked with a single argument.

if len (sys.argv) != 2:
    print ("Usage: %s <image-file>" % sys.argv[0], file=sys.stderr)
    exit (1)
# Process each file on the command line in turn.
for fn in sys.argv[1:]:
    img = cv2.imread (fn)
################################ Upload Here #################################
## This is the final assignment for CE866 Computer Vision. The task is to
## obtain the position and bearing of a red pointer on Horwood's handdrawn 
## map with a blue background. It also a green arrow that shows the direction 
## of north on the corner of the map. The tasks done below:
## 1. Segment the map from the blue background
## 2. Locate the green arrow and rotate it if not upright
## 3. Segment the red pointer
## 4. Locate the tip of the pointer to determine location
## 5. Determine orientation of the pointer, convert to bearing and output it

#################################### STEP 1 ####################################
######### remove blue background
# convert the input image to hsv space
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# Remove the blue background from the image while in 
# HSV colorspace by defining the upper and lower bounds of the color blue 
# and using opencv inRange function to convert all the values within that 
# range to black
# Masking is used to focus on the color of the image we're working 
upper_blue = np.array([135,255,255])  # upper range of color blue
lower_blue = np.array([85,50,0])  # lower range of color blue
mask = cv2.inRange(hsv, lower_blue, upper_blue)
res = cv2.bitwise_not(hsv, hsv, mask= mask)  # Contains grey pixels 
img[mask!=0] =[0,0,0]

#################################### STEP 2 ####################################

#This next step find the contours of the dark part of the image to segment 
# the map from the background.

# Read image and convert to grayscale, then apply binary thresholding 
# to get an idea of where the edges are located. Using a threshold value 
# that makes most of the map white is ideal because the border will be 
# detected easily.

# The next step is to find and draw the contours using the drawContours() 
# method to overlay the contours on the original image.
# convert to grayscale
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# apply binary thresholding
# Any pixel with a value greater than 10 will be set 
# to a value of 255 (white)
ret, img_thresh = cv2.threshold(img_gray, 10, 255, cv2.THRESH_BINARY)

# detect the contours on the binary image using cv2.CHAIN_APPROX_SIMPLE
contours, hierarchy = cv2.findContours(image=img_thresh, 
                                       mode=cv2.RETR_TREE, 
                                       method=cv2.CHAIN_APPROX_SIMPLE)

for c in contours:
    epsilon = 0.1 * cv2.arcLength(c, True)
    approx_rectangle = cv2.approxPolyDP(c, epsilon, True)
    if len(approx_rectangle) == 4:
        break
        
# draw contours on the original image
image_copy = img.copy()
cv2.drawContours(image=image_copy, contours=c, 
                 contourIdx=0, color=(0, 255, 0), 
                 thickness=2, lineType=cv2.LINE_AA)

#################################### STEP 3 ####################################

#Use perspective transform to reshape the map by providing the points on the 
# image from which to gather information and wrap the original image.

a = approx_rectangle[0][0]
b = approx_rectangle[1][0]
c = approx_rectangle[2][0]
d = approx_rectangle[3][0]
# draw the contour onto the map
img_pts = cv2.drawContours(img, approx_rectangle, -1, (0, 255, 0), 3)
# define the dimensions of the image to be transformed, length and width
# the perspective transform code was adapted from
# https://stackoverflow.com/questions/67962200/python-opencv-perspective-
# transformation-problems

wid_ad = np.sqrt(((a[1] - d[1])**2) + ((a[0] - d[0])**2))
wid_cb = np.sqrt(((c[1] - b[1])**2) + ((c[0] - b[0])**2))
len_dc = np.sqrt(((d[1] - c[1])**2) + ((d[0] - c[0])**2))
len_ab = np.sqrt(((a[1] - b[1])**2) + ((a[0] - b[0])**2))
# define the max length and width of the new image
max_width = max(int(wid_ad), int(wid_cb))
max_length = max(int(len_dc), int(len_ab))


# Locate points of the documents
# or object which you want to transform
# pts1 is the original coordinates of the image
# pts2 is the coordinates used to transform the image
# this part took some trial and error to find the proper positioning
pts1 = np.float32([b, c, d, a])

pts2 = np.float32([[0,0],
                   [0, max_width],
                   [max_length, max_width],
                   [max_length, 0]])

# Apply Perspective Transform Algorithm and pass it to the image
matrix = cv2.getPerspectiveTransform(pts1, pts2)
img_cropped = cv2.warpPerspective(img_pts, matrix, ( max_length,max_width))


#################################### STEP 4 ####################################

# Find the contour of the green arrow.
# finding the green contour runs inside this loop

run = 1
i = 0
while run:
    # convert again to hsv space
    hsv = cv2.cvtColor(img_cropped, cv2.COLOR_BGR2HSV)
    img_pts = 0
    # remove green from the image
    upper_green = np.array([75, 255, 255]) #-- upper range --
    lower_green = np.array([45, 100, 100])  #-- lower range --
    mask_green = cv2.inRange(hsv, lower_green, upper_green)

    # detect the contours on the binary image using cv2.CHAIN_APPROX_NONE
    contours, hierarchy = cv2.findContours(image=mask_green, 
                                           mode=cv2.RETR_TREE, 
                                           method=cv2.CHAIN_APPROX_NONE)
    #################
    # draw the contours around the arrow to see the points 
    img_pts = cv2.drawContours(img_cropped, contours, -1, (255, 0, 0), 3)

    # write a for loop to find the biggest contour
    for c in contours:
        epsilon = 0.1 * cv2.arcLength(c, True)
        approx_polygon = cv2.approxPolyDP(c, epsilon, True)
        if len(approx_polygon) > 3 and cv2.contourArea(c) < 1000:
            break
    arrow_region = approx_polygon[0][0]   
    
    if arrow_region[0] > 700 and arrow_region[1] < 500:
        run = 0   
    else:
        img_cropped = cv2.rotate(img_cropped, cv2.ROTATE_180)

#################################### STEP 5 ####################################

# Find the contour of the red pointer
# convert to hsv space
hsv = cv2.cvtColor(img_cropped, cv2.COLOR_BGR2HSV)

# remove red from the image
# red has two regions on the color spectrum
lower_red1 = np.array([0, 100, 30])    # upper range 
upper_red1 = np.array([30, 255, 255])  # lower range
lower_red2 = np.array([145, 100, 30])  # upper range
upper_red2 = np.array([225, 255, 255]) # lower range

# generate the lower and upper mask of red
mask_red1 = cv2.inRange(hsv, lower_red1, upper_red1)
mask_red2 = cv2.inRange(hsv, lower_red2, upper_red2)

# merge the masks
mask_red = cv2.bitwise_or(mask_red1, mask_red2)
    
# detect the contours on the binary image using cv2.CHAIN_APPROX_NONE
contours, hierarchy = cv2.findContours(image=mask_red, 
                                       mode=cv2.RETR_TREE, 
                                       method=cv2.CHAIN_APPROX_SIMPLE)

# write a for loop to find the contour
for c in contours:
    epsilon = 0.1 * cv2.arcLength(c, True)
    approx_triangle = cv2.approxPolyDP(c, epsilon, True)
    if len(approx_triangle) == 3:
        break

# draw contours on the original image
image_copy = img_cropped.copy()
cv2.drawContours(image=image_copy, contours=c, 
                 contourIdx=-1, color=(0, 255, 0), 
                 thickness=2, lineType=cv2.LINE_AA) 
   

                                                  
#################################### STEP 6 ####################################

# To find position and bearing
# The red pointer is an isosceles triangle, by finding the length
# of the sides, we can determine from the lengths with roughly equal 
# sides which one is the pointer and use that to find the position
# distance formula obtained from
# https://www.thoughtco.com/understanding-the-distance-formula-2312242
a = approx_triangle[0][0]
b = approx_triangle[1][0]
c = approx_triangle[2][0]

ab = np.sqrt(((a[1] - b[1])**2) + ((a[0] - b[0])**2))
bc = np.sqrt(((c[1] - b[1])**2) + ((c[0] - b[0])**2))
ca = np.sqrt(((c[1] - a[1])**2) + ((c[0] - a[0])**2)) 

# since the two sides of the triangle are roughly the same size and both 
# greater than 130,
# the smallest side is the opposite of the pointer
if ab > 100 and bc > 100:
    point = b
elif bc > 100 and ca > 100:
    point = c
elif ca > 100 and ab > 100:
    point = a
# now to find the midpoint    
if ab < 130:
    midpoint = (a+b)/2
elif bc < 130:
    midpoint = (b+c)/2
elif ca < 130:
    midpoint = (a+c)/2 

# to find the position of the pointer:
xposs = point[0]
yposs = point[1]

xpos = xposs/max_length
ypos = 1 - (yposs/max_width)

# To calculate bearing, we will use trigonometric functions and the
#  coordinates from the red pointer
y = midpoint
x = point
hdg_initial = np.arctan2(y[1]-x[1], y[0]-x[0])
hdg = np.degrees(hdg_initial)+270

##############################################################################    

# Output the position and bearing in the form required by the test harness.
print ("POSITION %.3f %.3f" % (xpos, ypos))
print ("BEARING %.1f" % hdg)
################    END   #####################################################
