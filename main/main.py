import cv2
import numpy as np
import planimeter as planimeter
import reference as ref
import tkinter as tk
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
   CHANGER - Permettez à l'utilisateur de cliquer sur les deux points de l'étalon (1 cm).
   CHANGER - Calculez la distance en pixels entre ces points.
   FAIT - Cliquer sur la geometrie de reference et calculer le nombre de pixels
   FAIT - Enregistrez cette valeur pour la conversion ultérieure de pixels en cm².

4. Conversion des pixels en cm² :
   FAIT - Lorsque l'utilisateur clique sur la géométrie, mesurez la distance entre les points en pixels.
   FAIT - Utilisez la relation entre la distance en pixels et la distance en cm pour convertir l'aire en cm².

5. Calcul de l'aire :
   FAIT - Lorsque l'utilisateur clique sur une géométrie, 
        mesurez la surface à l'aide des coordonnées de la géométrie et de la distance en pixels.

6. Affichage de l'aire :
   FAIT (mais doit optimiser) - Affichez l'aire calculée dans la console ou sur l'interface graphique.
"""


# EVENEMENT SUIVIES


EVENT_REFERENCE_START = True
EVENT_REFERENCE_DONE = False
EVENT_PLANIMETER_MESUREMENT = False


# VALEUR GLOBALE


REF_POS = [0, 0, 0, 0]
REF_DENSITY = None


# EVENEMENT SOURIS


def mouse_callback(event, x, y, flags, param):
    global EVENT_REFERENCE_START, EVENT_REFERENCE_DONE, EVENT_PLANIMETER_MESUREMENT, \
        REF_POS, scanner_image, REF_DENSITY

    range_colorval = 20

    # Méthode avec click et utilisation algo_planimeter classique
    if ref.set_reference(event, x, y, flags, param) and not EVENT_REFERENCE_DONE:
        REF_DENSITY = ref.mm_area_of_pixel_unit_with_counts_know(
            len(planimeter.surface_area(x, y, range_colorval, IMAGE_ARRAY, False)))
        EVENT_REFERENCE_DONE = True
        print("Reference calculé")
        return None
    # Méthode avec rectangle formé par 2 points
    """
    if not EVENT_REFERENCE_DONE:
        status = ref.draw_rectangle_evenement(event, x, y, flags, param)
        if event == cv2.EVENT_LBUTTONDOWN and not status[2] and EVENT_REFERENCE_START:
            REF_POS[0] = status[0]
            REF_POS[1] = status[1]
            EVENT_REFERENCE_START = False
            return None
        elif status[2]:
            REF_POS[2] = status[0]
            REF_POS[3] = status[1]
            # print("reference affiché")
            # print(REF_POS)
            cv2.rectangle(scanner_image, (REF_POS[0], REF_POS[1]), (REF_POS[2], REF_POS[3]), (0, 0, 0), -1)
            REF_DENSITY = ref.mm_area_of_pixel_unit(REF_POS)
            # print("l'unité de réference sera --> ", ref.mm_area_of_pixel_unit(REF_POS), "mm² par pixel")
            # print("l'unité de réference sera --> ", ref.mm_per_pixel_unit(REF_POS), "mm par pixel")
            EVENT_REFERENCE_DONE = True
            return None
        return None
    """
    if event == cv2.EVENT_LBUTTONDOWN:
        # print(f"Mouse clicked at position: ({x}, {y})")
        # print(IMAGE_ARRAY[y][x])
        pixel_list = planimeter.surface_area(x, y, range_colorval, IMAGE_ARRAY, False)

        print("LA REFERENCE A ETE CALCULE ---> ", REF_DENSITY, "mm²/pixel (BIEN CALCULER)")
        # print("l'aire est de :", planimeter.surface_area(x, y, range_colorval, IMAGE_ARRAY))
        # print("l'aire est de :", REF_DENSITY*pixel_area, "mm²")
        print("l'aire est de :", round(REF_DENSITY*len(pixel_list)/100, 2), "cm²")
        planimeter.display_surface_info(planimeter.info_from_surface(pixel_list, REF_DENSITY))


# Debut code creation IMAGE


scanner_image = cv2.imread('../scan_home/100_PPP.png')
IMAGE_ARRAY = np.array(scanner_image)
BOOL_ARRAY = np.empty(IMAGE_ARRAY.shape)
gray_image = cv2.cvtColor(scanner_image, cv2.COLOR_BGR2GRAY)
#       cv2.imshow('GrayImage', gray_image)# cv2.setMouseCallback('GrayImage', mouse_callback)
# cv2.imshow('Image', scanner_image)
# cv2.setMouseCallback('Image', mouse_callback)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
while True:
    cv2.imshow('Image', scanner_image)
    cv2.setMouseCallback('Image', mouse_callback)
    if cv2.waitKey(10) == 27:
        break
cv2.destroyAllWindows()
