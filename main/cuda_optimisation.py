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
def test(an_array, colour, range_val):
    # procedure de base qui permet d'être dans les limites de l'image
    x, y = cuda.grid(2)
    if x < an_array.shape[0] and y < an_array.shape[1]:
        # an_array[x][y] = 1
        temp_arr = an_array[x][y]

        # Condition de test
        for i in range(len(temp_arr)):
            if colour[i]+range_val > temp_arr[i] > colour[i]-range_val:
                # an_array[x][y] = 255
                continue
            else:
                an_array[x][y] = 0
                break
        pass
    pass
    """
    if np.all(rmin <= colour) and np.all(colour <= rmax):
        image_array = True
    else:
        image_array = False
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
    image = image_array #cv2.imread(image_path)
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
def bfs_cuda_algorithm(an_array, precolor, range_val, neighboor_list):
    # procedure de base qui permet d'être dans les limites de l'image
    x, y = cuda.grid(2)
    cond_1 = x < an_array.shape[0] and y < an_array.shape[1]
    cond_2 = neighboor_list.any([x, y])
    # on vérifie si les coordonées sont voisines et sont connues
    if cond_1 and cond_2:
        for k in range(-1, 2):
            for m in range(-1, 2):
                # an_array[x][y] = 1
                temp_arr = an_array[x+k][y+m]
                # Condition de test
                is_same_color = True
                for i in range(len(temp_arr)):
                    if precolor[i]+range_val > temp_arr[i] > precolor[i]-range_val:
                        # an_array[x][y] = 255
                        continue
                    else:
                        is_same_color = False
                        an_array[x][y] = 0
                        break
                if is_same_color:
                    neighboor_list.append([x+k, y+m])


def init_cuda_optimisation(image_array, x_init, y_init, range_val):
    # threadsperblock = 32
    # blockspergrid = (image_array.size + (threadsperblock - 1)) // threadsperblock

    data = np.copy(image_array)
    precolor = data[y_init][x_init]

    # Application de cuda sur un tableau à 2D dimension
    # INITIALISATION GENERIQUE POUR LA 2D !!!!!
    # INITIALISATION GENERIQUE POUR LA 2D !!!!!
    # INITIALISATION GENERIQUE POUR LA 2D !!!!!

    # threadsperblock = (16, 16)
    # blockspergrid_x = math.ceil(image_array.shape[0] / threadsperblock[0])
    # blockspergrid_y = math.ceil(image_array.shape[1] / threadsperblock[1])
    # blockspergrid = (blockspergrid_x, blockspergrid_y)
    # init_list = np.array([x_init, y_init])
    # test[blockspergrid, threadsperblock](image_array, precolor, range_val)

    # INITIALISATION GENERIQUE POUR LA 2D !!!!!
    # INITIALISATION GENERIQUE POUR LA 2D !!!!!
    # INITIALISATION GENERIQUE POUR LA 2D !!!!!

    neighboor_list = np.array([x_init, y_init])
    while len(neighboor_list) > 0:
        threadsperblock = (16, 16)
        blockspergrid_x = math.ceil(image_array.shape[0] / threadsperblock[0])
        blockspergrid_y = math.ceil(image_array.shape[1] / threadsperblock[1])
        blockspergrid = (blockspergrid_x, blockspergrid_y)
        bfs_cuda_algorithm[blockspergrid, threadsperblock](image_array, precolor, range_val, neighboor_list)
    """
    verify_condition_neighboor[blockspergrid, threadsperblock]([x_init, y_init],
                                                               rmin, precolor, rmax,
                                                               image_array.shape[0], image_array.shape[1],
                                                               out_array)
    """


@cuda.jit
def search_algorithm(neighboor_list, precolor, range_val, an_array):
    i = cuda.grid(1)
    if i < neighboor_list.size:
        actual_cell = neighboor_list[i]
        for j in range(len(actual_cell)):
            actual_cell[j] = j
        # for j in range(len(neighboor_list[i])):
        #     x = neighboor_list[i][j]
        #     y = neighboor_list[i][j]
    """
    if pos < neighboor_list.size:
        bool_1 = 0 < pos[0][0] < an_array.shape[1]
        bool_2 = 0 < pos[0][1] < an_array.shape[0]
        if bool_1 and bool_2:
            x = pos[0]
            y = pos[1]
            for k in range(-1, 2):
                for m in range(-1, 2):
                    # an_array[x][y] = 1
                    temp_arr = an_array[x + k][y + m]
                    # Condition de test
                    is_same_color = True
                    for i in range(len(temp_arr)):
                        if precolor[i] + range_val > temp_arr[i] > precolor[i] - range_val:
                            # an_array[x][y] = 255
                            continue
                        else:
                            is_same_color = False
                            an_array[x][y] = 0
                            break
                    if is_same_color:
                        neighboor_list.append([x + k, y + m])
        neighboor_list.pop()
    """


def init_algo_search(image_array, x_init, y_init, range_val):
    data = np.copy(image_array)
    precolor = data[y_init][x_init]
    neighboor_list = np.array([x_init, y_init], dtype=np.int64)
    while len(neighboor_list) > 0:
        threadsperblock = 32
        blockspergrid = (neighboor_list.size + (threadsperblock - 1)) // threadsperblock
        search_algorithm[blockspergrid, threadsperblock](neighboor_list, precolor, range_val, image_array)
        print(neighboor_list)


# CE BOUT DE CODE EN DESSOUS PERMET DE TESTER DIRECTEMENT POUR VOIR SI MA FONCTION MARCHE
# AVANT DE L'IMPLEMENTER SUR PLANIMETER.PY
"""
import cv2


scanner_image = cv2.imread('../scan_home/100_PPP.png')


def mouse_callback(event, x, y, flags, param):
    global scanner_image

    if event == cv2.EVENT_LBUTTONDOWN:
        image_array = np.array(scanner_image)
        # print("AVANT CUDA")
        # print(IMAGE_ARRAY)


        # init_cuda_optimisation(image_array, x, y, 30)
        init_algo_search(image_array, x, y, 30)
        print("THIS SHAPE --> ", image_array.shape[0], image_array)
        print("APRES CUDA")
        # permet de tout afficher ---> np.set_printoptions(threshold=np.inf)
        print(image_array)
        cv2.imshow('resultat', image_array)
"""

"""
cv2.imshow('Image', scanner_image)
cv2.setMouseCallback('Image', mouse_callback)
cv2.waitKey(0)
cv2.destroyAllWindows()
"""
