import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import imutils
import warnings

warnings.filterwarnings('ignore')

clusters = 5

img = cv2.imread('Test_img.jpg')
org_img = img.copy()
print(f'Original img shape: {img.shape}')

kmg = imutils.resize(img, height=200, width=300)
print(f'Resized img shape: {img.shape}')

flat_img = np.reshape(img, (-1,3))
print(f'Flat img shape: {flat_img.shape}')

kmeans = KMeans(n_clusters=clusters)
kmeans.fit(flat_img)

dom_colors = np.array(kmeans.cluster_centers_, dtype= 'uint')

percentage = (np.unique(kmeans.labels_, return_counts= True)[1])/flat_img.shape[0]
p_and_c = zip(percentage, dom_colors)
p_and_c = sorted(p_and_c, reverse= True)

block = np.ones((50,50,3), dtype= 'uint')
plt.figure(figsize= (12, 8))
for i in range(clusters):
    plt.subplot(1, clusters, i+1)
    block[:] = p_and_c[i][1][::-1]
    plt.imshow(block)
    plt.xticks([])
    plt.yticks([])
    plt.xlabel(str(round(p_and_c[i][0]*100,2))+'%')

rows = 1000
cols = int((org_img.shape[0]/org_img.shape[1])*rows)
img = cv2.resize(org_img,dsize=(rows,cols), interpolation=cv2.INTER_LINEAR)

copy = img.copy()
cv2.rectangle(copy, (rows//2-250, cols//2-90), (rows//2+250, cols//2+110), (255,255,255), -1)

final = cv2.addWeighted(img,0.1,copy,0.9,0)
cv2.putText(final, 'Most Dominant Colors in the Image', (rows//2-230, cols//2-40), cv2.FONT_HERSHEY_DUPLEX, 0.8, (0,0,0), 2, cv2.LINE_AA)

start = rows//2-220
for i in range(5):
    end = start + 70
    final[cols//2:cols//2+70, start:end] = p_and_c[i][1]
    cv2.putText(final,str(i+1), (start+25, cols//2+45), cv2.FONT_HERSHEY_DUPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
    start = end+20

plt.show()

cv2.imshow('img', final)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite('output.jpg', final)