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
        REF_POS, scanner_image, REF_DENSITY, IMAGE_PATH

    range_colorval = 10
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
    # Indique le dernier endroit qui a été cliqué
    # if event == cv2.EVENT_RBUTTONDOWN:
    #    cv2.setWindowTitle('Image', 'x:{0} y:{1}'.format(x, y))
    if event == cv2.EVENT_LBUTTONDOWN:
        # print(f"Mouse clicked at position: ({x}, {y})")
        # print(IMAGE_ARRAY[y][x])
        pixel_list = planimeter.surface_area(x, y, range_colorval, IMAGE_ARRAY, False)

        print("LA REFERENCE A ETE CALCULE ---> ", REF_DENSITY, "mm²/pixel (BIEN CALCULER)")
        # print("l'aire est de :", planimeter.surface_area(x, y, range_colorval, IMAGE_ARRAY))
        # print("l'aire est de :", REF_DENSITY*pixel_area, "mm²")
        print("l'aire est de :", round(REF_DENSITY*len(pixel_list)/100, 2), "cm²")

        cv2.imshow('Area selectionned', planimeter.draw_foundedarea(IMAGE_PATH, pixel_list, False))
        # je dessine d'abord car après quand la fenêtre est ouverte, l'application est focus sur cette fenêtre
        planimeter.display_surface_info(planimeter.info_from_surface(pixel_list, REF_DENSITY))


# Debut code creation IMAGE

IMAGE_PATH = '../scan_home/100_PPP.png'
scanner_image = cv2.imread('../scan_home/100_PPP.png')
IMAGE_ARRAY = np.array(scanner_image)
BOOL_ARRAY = np.empty(IMAGE_ARRAY.shape)
gray_image = cv2.cvtColor(scanner_image, cv2.COLOR_BGR2GRAY)


image = scanner_image

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Détection des zones bleues (bic bleu)
# SUR OPENCV
# HSV - H : de 0 a 180 | S : 0 a 255 | V : 0 a 255
# EN Standard
# HSV - normal va de H 0-360deg |S en % | V en %
# lower_blue = np.array([90, 50, 50])  # Définir la plage de valeurs bleues inférieures
lower_blue = np.array([0, 2, 0])  # Définir la plage de valeurs bleues inférieures
upper_blue = np.array([180, 255, 255])  # Définir la plage de valeurs bleues supérieures
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)

# Détection des zones noires (couleur noire)
lower_black = np.array([0, 0, 0])  # Définir la plage de valeurs noires inférieures
upper_black = np.array([360, 255, 40])  # Définir la plage de valeurs noires supérieures
mask_black = cv2.inRange(hsv, lower_black, upper_black)

# Combiner les masques bleus et noirs
mask_combined = cv2.bitwise_or(mask_blue, mask_black)

# Appliquer le masque pour conserver les parties bleues et noires
result = cv2.bitwise_and(image, image, mask=mask_combined)

# Remplacer le reste par du blanc
result[np.where((result == [0, 0, 0]).all(axis=2))] = [255, 255, 255]









# TRAITEMENT POUR AVOIR UN FOND BLANC
# equ = cv2.equalizeHist(gray_image)
# equ_color = cv2.cvtColor(equ, cv2.COLOR_GRAY2BGR)
# IMAGE_ARRAY = equ_color

#       cv2.imshow('GrayImage', gray_image)# cv2.setMouseCallback('GrayImage', mouse_callback)
# cv2.imshow('Image', scanner_image)
# cv2.setMouseCallback('Image', mouse_callback)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
while True:
    # cv2.imshow('Image', scanner_image)
    # avec np.hstack --> permet d'afficher 2 image à la fois
    # cv2.imshow('Image', np.hstack((scanner_image, equ_color)))
    cv2.imshow('Image', result)
    cv2.setMouseCallback('Image', mouse_callback)
    if cv2.waitKey(10) == 27:
        break
cv2.destroyAllWindows()
