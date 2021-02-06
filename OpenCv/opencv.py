import cv2
img = cv2.imread('C:\\Users\\asus\\Desktop\\Knn\\KNN\\datasets\\TezYemek\\KabakTatlisi\\img98.png', cv2.IMREAD_UNCHANGED)

print('Original Dimensions : ', img.shape)

width = 100
height = 50  # keep original height
dim = (width, height)

# resize image
resized = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
print('Resized Dimensions : ', resized.shape)
cv2.imshow("Resized image", img)

cv2.waitKey(0)