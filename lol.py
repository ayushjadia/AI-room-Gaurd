import cv2
img = cv2.imread("enroll_images/Ayush_1.jpg")
print(img.shape, img.dtype)
cv2.imshow("Check", img)
cv2.waitKey(0)