import cv2
import numpy as np

img_paths = ["./img/Sala1.jpg", "./img/Sala2.jpg", "./img/Sala3.jpeg"]

# HARRIS
def Harris(imagem_path, bS, Ks, K, window_name):
    imagem = cv2.imread(imagem_path, cv2.IMREAD_GRAYSCALE)
    
    # Algoritmo usado para detectar cantos - a detecção ocorre com uma mudança de pixel em multiplas direções

    # blockSize = Tamanho do vizinho considerado para a detecção do canto(padrao:2)
    # ksize = Tamanho da abertura do sobel para derivadas(padrao:3)
    # k = Parametro livre da equação de harris(padrao:0.04)
    harris = cv2.cornerHarris(imagem, blockSize=bS, ksize=Ks, k=K)
    harris = cv2.dilate(harris, None)

    # Destacar os cantos na imagem original
    imagem[harris > 0.01 * harris.max()] = [255]

    # Exibir resultados
    cv2.imshow(window_name, imagem)

parametros = [
    (2, 3, 0.04),
    (5, 1, 0.2),
    (5, 9, 0.015)
]

for idx, img_path in enumerate(img_paths):
    for jdx, (bS, Ks, K) in enumerate(parametros):
        window_name = f'Image {idx+1} - Harris {jdx+1}'
        Harris(img_path, bS, Ks, K, window_name)

cv2.waitKey(0)
cv2.destroyAllWindows()
