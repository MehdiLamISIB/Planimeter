import cv2


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
