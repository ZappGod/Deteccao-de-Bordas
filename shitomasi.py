import cv2
import numpy as np

img_paths = ["./img/Sala1.jpg", "./img/Sala2.jpg", "./img/Sala3.jpeg"]


def ShiTomasi(imagem_path,maxC,Qlevel,mDistance,window_name):
    imagem = cv2.imread(imagem_path, cv2.IMREAD_GRAYSCALE)

    # Aplicar o detector Shi-Tomasi
    cantos = cv2.goodFeaturesToTrack(imagem, maxCorners=maxC, qualityLevel=Qlevel, minDistance=mDistance)
    cantos = np.intp(cantos)

    # Marcar os cantos na imagem
    for canto in cantos:
        x, y = canto.ravel()
        cv2.circle(imagem, (x, y), 3, 255, -1)
        
    cv2.imshow(window_name, imagem)

parametros = [(100,0.01,10),
              (50,0.05,5),
              (25,0.1,20)]

for idx, img_path in enumerate(img_paths):
    for jdx, (maxC, Qlevel, mDistance) in enumerate(parametros):
        window_name = f'Image {idx+1} - Shi-Tomasi {jdx+1}'
        ShiTomasi(img_path, maxC, Qlevel, mDistance, window_name)

cv2.waitKey(0)
cv2.destroyAllWindows()