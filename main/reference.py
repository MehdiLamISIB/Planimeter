import cv2
import math
# PPI --> pixel per inch
# PPP --> pixel par pouce
# DPI --> dot (encre) per inch
# https://fr.wikipedia.org/wiki/R%C3%A9solution_spatiale_des_images_matricielles#Pixel_par_pouce
"""
Pour une page A4 avec ppp=300
- taille en pixels : 2 483 x 3 502
- taille en mm : 210 x 297
- taille du fichier : 25 Mo
"""


INCH_TO_CM = 2.54
CM_SQUARED_TO_MM_SQUARED = 100


def ppi_to_pixel_number(x, y, ppp): return (x*ppp/INCH_TO_CM)*(y*ppp/INCH_TO_CM)


def rescale_image(image, xres, yres):
    """
    Permet de mettre change la résolution de l'image affiché
    :param image: image à afficher
    :param xres: diviseur en x (ex. xres=2 --> x=x/2)
    :param yres: diviseur en y
    :return:
    """
    scale_percent = 60  # percent of original size
    width = int(image.shape[1] * scale_percent / xres)
    height = int(image.shape[0] * scale_percent / yres)
    dim = (width, height)
    image = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
    return image


# draw_rectangle_evenement : JE NE VAIS PAS UTILISER CELA, JE VAIS CLIQUER ET PUIS CA VA DONNER PIXEL DANS 1cm²


def draw_rectangle_evenement(event, x, y, flags, param):
    """
    Permet de dessiner la référence de 1cm et de récuper la diagonale
    :param event: event de la fenêtre openCV
    :param x: position x de la souris
    :param y: position y de la souris
    :param flags:
    :param param:
    :return: [position x, position y, référence faites]
    """
    if event == cv2.EVENT_LBUTTONDOWN:
        return [x, y, False]
    if event == cv2.EVENT_LBUTTONUP:
        return [x, y, True]
    else:
        return [x, y, None]


# set_reference: EVENT N°1 pour meilleur lisibilité


def set_reference(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        return True
    else:
        return False


# FONCTION APPLIQUE SI UTILISATION AVEC 4 COINS ET 2 CLICK SOURIS POUR FAIRE CARREE
# pixel_per_cm : retourne le nombre de pixel dans 1cm² ce qui donne la densité pixel/cm²


def pixels_per_cm_squared(ref_pos): return abs(ref_pos[2]-ref_pos[0])*abs(ref_pos[3]-ref_pos[1])


# mm_squared_per_pixel_unit : donne le nombre de pixel par mm²


def pixels_per_mm_squared(ref_pos): return pixels_per_cm_squared(ref_pos)/CM_SQUARED_TO_MM_SQUARED


# mm_area_of_pixel_unit : donne l'aire d'un pixel en mm² (1mm²/(nombre de pixel dans 1mm²))


def mm_area_of_pixel_unit(ref_pos): return 1/pixels_per_mm_squared(ref_pos)


# mm_per_pixel_unit : donne la longueur d'un pixel en mm


def mm_per_pixel_unit(ref_pos): return math.sqrt(mm_area_of_pixel_unit(ref_pos))


# mm_area_of_pixel_unit_with_counts_know: donne l'aire en connaisant le nombre de pixels (pas de calculs en plus)


def mm_area_of_pixel_unit_with_counts_know(pixels_count): return 1/(pixels_count/CM_SQUARED_TO_MM_SQUARED)
