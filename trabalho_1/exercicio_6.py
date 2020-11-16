
# Python program to illustrate 
# multiscaling in template matching 
import cv2 
import numpy as np 
import imutils

 

# Read the template 
template = cv2.imread('babooneye.png')
template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
template = cv2.Canny(template, 50, 200)
(tH, tW) = template.shape[:2]

# Read the main image 
img_rgb = cv2.imread('baboon.png') 
img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)

found = None

for scale in np.linspace(0.2, 1.0, 20)[::-1]: 

    resized = imutils.resize(img_gray, width = int(img_gray.shape[1] * scale)) 
    r = img_gray.shape[1] / float(resized.shape[1]) 
    
    if resized.shape[0] < tH or resized.shape[1] < tW: 
        break

    borda = cv2.Canny(resized, 50, 200)
    result = cv2.matchTemplate(borda, template, cv2.TM_CCOEFF_NORMED)
    (_, maxVal, _, maxLoc) = cv2.minMaxLoc(result)

    clone = np.dstack([borda, borda, borda])
    cv2.rectangle(clone, (maxLoc[0], maxLoc[1]),
	(maxLoc[0] + tW, maxLoc[1] + tH), (0, 0, 255), 2)
    cv2.imshow("Visualize", clone)
    cv2.waitKey(0)

    found = (maxVal, maxLoc, r)
# unpack the found varaible and compute the (x, y) coordinates 
# of the bounding box based on the resized ratio 
(_, maxLoc, r) = found 
(startX, startY) = (int(maxLoc[0] * r), int(maxLoc[1] * r)) 
(endX, endY) = (int((maxLoc[0] + tW) * r), int((maxLoc[1] + tH) * r)) 

# draw a bounding box around the detected result and display the image 
cv2.rectangle(img_rgb, (startX, startY), (endX, endY), (0, 0, 255), 2) 
cv2.imshow("Image", img_rgb) 
cv2.waitKey(0) 