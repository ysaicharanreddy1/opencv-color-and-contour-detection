import cv2
import cvzone
from cvzone.Utils import findContours
from cvzone.ColorModule import ColorFinder
cap = cv2.VideoCapture(0)
myColorFinder = ColorFinder(trackBar=False)
# Yellow HSV
yellowHSV = {
    'hmin': 21,
    'smin': 70,
    'vmin': 0,
    'hmax': 56,
    'smax': 255,
    'vmax': 255
}
# Red HSV
redHSV = {
    'hmin': 0,
    'smin': 135,
    'vmin': 161,
    'hmax': 48,
    'smax': 255,
    'vmax': 255
}
# Blue HSV
blueHSV = {
    'hmin': 73,
    'smin': 79,
    'vmin': 89,
    'hmax': 103,
    'smax': 255,
    'vmax': 255
}
# Green HSV
greenHSV = {
    'hmin': 56,
    'smin': 71,
    'vmin': 35,
    'hmax': 78,
    'smax': 255,
    'vmax': 255
}
# Select the color you want to detect
hsvVal = blueHSV
# hsvVal = redHSV
# hsvVal = blueHSV
# hsvVal = greenHSV
while True:
    success, img = cap.read()
    if not success:
        break
    imgColor, mask = myColorFinder.update(img, hsvVal)
    imgcontours, confound = findContours(img, mask, filter = None)
    imgStack = cvzone.stackImages([img, imgColor, mask, imgcontours], 4, 0.5)
    cv2.imshow("Color Detection", imgStack)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()