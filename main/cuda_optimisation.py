from numba import cuda
import numpy as np
import math

"""
Pour ce code, je vais devoir :
- prendre en compte que mon tableau de valeur grandira au fur est à mesure
- on commence avec une cellule qui a 4 voisines
- puis ces 4 voisines on 4 voisines chacunes, et ça évolue de façon exponentionnelle
- y=4^x
- ça sera une fonction en recursion qui s'arrête quand aucune valeur est voisines (bord)
- en input : les nouvelles voisines,
"""


class Tree:
    def __init__(self, x, y):
        self.startNode = self.Node(x, y)

    class Node:
        def __init__(self, x, y):
            self.x = x
            self.y = y
            self.left = None
            self.right = None
            self.up = None
            self.down = None

    def set_value_in_domain(self, x, y, xmin, ymin, xmax, ymax):
        node = self.Node(x, y)
        if x > xmin:
            node.left = x - 1
        if x < xmax:
            node.right = x + 1
        if y < ymax:
            node.down = x + 1
        if y > ymin:
            node.up = x - 1
        return node


@cuda.jit
def test(an_array, rmin, colour, rmax):
    x, y = cuda.grid(2)
    if x < an_array.shape[0] and y < an_array.shape[1]:
        # an_array[x, y] += 1
        an_array[x][y] = 0
        pass
    pass
    """
    if np.all(rmin <= colour) and np.all(colour <= rmax):
        image_array = True
    else:
        image_array = False
    """


@cuda.jit
def verify_condition_neighboor(neighboor_array, rmin, colour, rmax, border_x, border_y, out_arr):
    # out_arr = np.empty(shape=(1, 0))
    pos = np.array([[0, 1], [0, -1], [1, 0], [-1, 0]])
    for i in range(len(neighboor_array)):
        for j in range(4):
            x = neighboor_array[i][0] + pos[j][0]
            y = neighboor_array[i][1] + pos[j][1]

            bool_col = np.all(rmin <= colour) and np.all(colour <= rmax)
            bool_border = (x < 0 or y < 0) or (y >= border_y or x >= border_x)

            if bool_col and not bool_border:
                np.append(out_arr, [x, y])


@cuda.jit
def bfs_parallel_implementation():
    pass


def init_cuda_optimisation(image_array, x_init, y_init, range_val):
    # threadsperblock = 32
    # blockspergrid = (image_array.size + (threadsperblock - 1)) // threadsperblock
    out_array = np.empty(shape=(1, 0))

    data = np.copy(image_array)
    precolor = data[y_init][x_init]
    rmin = precolor - np.array([range_val, range_val, range_val])
    rmax = precolor + np.array([range_val, range_val, range_val])

    # Application de cuda sur un tableau à 2D dimension
    # INITIALISATION GENERIQUE POUR LA 2D !!!!!
    # INITIALISATION GENERIQUE POUR LA 2D !!!!!
    # INITIALISATION GENERIQUE POUR LA 2D !!!!!
    threadsperblock = (16, 16)
    blockspergrid_x = math.ceil(image_array.shape[0] / threadsperblock[0])
    blockspergrid_y = math.ceil(image_array.shape[1] / threadsperblock[1])
    blockspergrid = (blockspergrid_x, blockspergrid_y)
    test[blockspergrid, threadsperblock](image_array, rmin, precolor, rmax)
    # INITIALISATION GENERIQUE POUR LA 2D !!!!!
    # INITIALISATION GENERIQUE POUR LA 2D !!!!!
    # INITIALISATION GENERIQUE POUR LA 2D !!!!!


    """
    verify_condition_neighboor[blockspergrid, threadsperblock]([x_init, y_init],
                                                               rmin, precolor, rmax,
                                                               image_array.shape[0], image_array.shape[1],
                                                               out_array)
    """


# CE BOUT DE CODE EN DESSOUS PERMET DE TESTER DIRECTEMENT POUR VOIR SI MA FONCTION MARCHE
# AVANT DE L'IMPLEMENTER SUR PLANIMETER.PY


import cv2
scanner_image = cv2.imread('../scan_home/100_PPP.png')
IMAGE_ARRAY = np.array(scanner_image)


# print("AVANT CUDA")
# print(IMAGE_ARRAY)
init_cuda_optimisation(IMAGE_ARRAY, 30, 50, 30)
print("THIS SHAPE --> ", IMAGE_ARRAY.shape[0], IMAGE_ARRAY)
print("APRES CUDA")
# permet de tout afficher ---> np.set_printoptions(threshold=np.inf)
print(IMAGE_ARRAY)