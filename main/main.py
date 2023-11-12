import cv2
import numpy as np
import planimeter as planimeter
import reference as ref


"""
Projet :
Cela semble être un projet intéressant et un peu complexe, mais nous pouvons le diviser en étapes plus petites. 
Voici une approche générale que vous pouvez suivre pour créer votre planimètre en Python :

1. Chargement de l'image :
   FAIT - Utilisez une bibliothèque comme OpenCV ou PIL pour charger l'image.
   FAIT - Affichez l'image dans une fenêtre pour que l'utilisateur puisse sélectionner l'étalon.

2. Traitement de l'image pour délimiter les contours :
   - Appliquez des filtres d'amélioration d'image pour améliorer la qualité de l'image,
        compte tenu des imperfections dues au scanner.
   - Utilisez des techniques de détection de contours (par exemple, Canny dans OpenCV) 
        pour extraire les contours de la géométrie.

3. Sélection de l'étalon :
   - Permettez à l'utilisateur de cliquer sur les deux points de l'étalon (1 cm).
   - Calculez la distance en pixels entre ces points.
   - Enregistrez cette valeur pour la conversion ultérieure de pixels en cm².

4. Conversion des pixels en cm² :
   - Lorsque l'utilisateur clique sur la géométrie, mesurez la distance entre les points en pixels.
   - Utilisez la relation entre la distance en pixels et la distance en cm pour convertir l'aire en cm².

5. Calcul de l'aire :
   - Lorsque l'utilisateur clique sur une géométrie, 
        mesurez la surface à l'aide des coordonnées de la géométrie et de la distance en pixels.

6. Affichage de l'aire :
   - Affichez l'aire calculée dans la console ou sur l'interface graphique.
"""


# EVENEMENT SOURIS


def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        print(f"Mouse clicked at position: ({x}, {y})")
        print(IMAGE_ARRAY[y][x])
        range_colorval = 20
        print("l'aire est de :", planimeter.surface_area(x, y, range_colorval, IMAGE_ARRAY))


# Debut code creation IMAGE


scanner_image = cv2.imread('../scan_home/100_PPP.png')
IMAGE_ARRAY = np.array(scanner_image)
BOOL_ARRAY = np.empty(IMAGE_ARRAY.shape)
gray_image = cv2.cvtColor(scanner_image, cv2.COLOR_BGR2GRAY)
# cv2.imshow('GrayImage', gray_image)
cv2.imshow('Image', scanner_image)
# cv2.setMouseCallback('GrayImage', mouse_callback)
cv2.setMouseCallback('Image', mouse_callback)
cv2.waitKey(0)
cv2.destroyAllWindows()
