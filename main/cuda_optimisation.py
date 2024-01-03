from numba import cuda
import numpy as np
import math
import cv2

"""
Pour ce code, je vais devoir :
- prendre en compte que mon tableau de valeur grandira au fur est à mesure
- on commence avec une cellule qui a 4 voisines
- puis ces 4 voisines on 4 voisines chacunes, et ça évolue de façon exponentionnelle
- y=4^x
- ça sera une fonction en recursion qui s'arrête quand aucune valeur est voisines (bord)
- en input : les nouvelles voisines,
"""

# change_color_kernel : permet de remplir les pixels de l'aire calculé


@cuda.jit
def change_color_kernel(image, coordinates):
    x, y = cuda.grid(2)
    if x < image.shape[0] and y < image.shape[1]:
        for coord in coordinates:
            if x == coord[0] and y == coord[1]:
                image[x, y] = (0, 0, 0)  # Change pixel color to black


# change_color : appelle le kernel cuda et retourne la nouvelle image crée


def change_color(image_array, coordinates, show_traited_image):
    image = np.copy(image_array) # cv2.imread(image_path)
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert image to RGB

    threadsperblock = (16, 16)
    blockspergrid_x = (image.shape[0] + threadsperblock[0] - 1) // threadsperblock[0]
    blockspergrid_y = (image.shape[1] + threadsperblock[1] - 1) // threadsperblock[1]
    blockspergrid = (blockspergrid_x, blockspergrid_y)

    d_image = cuda.to_device(image)
    d_coordinates = cuda.to_device(np.array(coordinates))

    change_color_kernel[blockspergrid, threadsperblock](d_image, d_coordinates)
    d_image.copy_to_host(image)

    if show_traited_image:
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        equ = cv2.equalizeHist(gray_image)
        return cv2.cvtColor(equ, cv2.COLOR_GRAY2BGR)
    else:
        resize_img = cv2.resize(image, (0, 0), fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
        return cv2.cvtColor(resize_img, cv2.COLOR_RGB2BGR)  # Convert image back to BGR for OpenCV


@cuda.jit
def check_neighboor_cuda(data, vis, visited, colmin, colmax):
    i, j = cuda.grid(2)
    if i < visited.shape[0] and j < visited.shape[1]:
        pass


def bfs_cuda_jit(data, vis, obj, visited, colmin, colmax):
    past_visited = np.array([])

    next_move = 0
    # chaque mouvement est enregistre sur un byte, on change chaque bit pour définir les positions possible
    possible_move = np.copy([0]*visited)
    while len(past_visited) != next_move:
        past_visited = np.copy(visited)

        # transfer donnée au gpu
        d_data = cuda.to_device(data)
        d_vis = cuda.to_device(vis)
        d_obj = cuda.to_device(obj)
        d_visited = cuda.to_device(visited)

        # on va modifier cette table pour chaque valeur de past_visited

        # Define CUDA grid and block dimensions
        threadsperblock = (16, 16)  # Choose appropriate block size
        blockspergrid_x = (past_visited.shape[0] + threadsperblock[0] - 1) // threadsperblock[0]
        blockspergrid_y = (past_visited.shape[1] + threadsperblock[1] - 1) // threadsperblock[1]
        blockspergrid = (blockspergrid_x, blockspergrid_y)
        check_neighboor_cuda[blockspergrid, threadsperblock](
            d_data, d_vis, d_obj, d_visited, colmin, colmax
        )

