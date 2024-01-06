from numba import cuda, int32
from numba import jit
from numba import njit
import numpy as np
import cv2


STATIC_CALL_INCREMENT = 0

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


# @jit --> permet d'accelerer le code dans le CPU
# a utiliser quand on a pas idée du nombre de thread à distribuer (fonction avec tableau qui change)

# @jit(nopython=True, cache=True)

# @jit(nopython=True, cache=True)


@njit(cache=True)
def bfs_jit_parallell(obj, visited, vis, colmax, colmin, data, n, m):
    moves = np.array([
        [0, 1], [1, 0],
        [-1, 0], [0, -1]
    ])

    while obj.shape[0] > 0:
        coord = obj[0]
        x, y = coord[0], coord[1]

        obj = obj[1:]  # Retirer le premier élément de obj

        for pos in moves:
            new_x, new_y = x + pos[0], y + pos[1]
            if new_x < 0 or new_y < 0 or new_x >= m or new_y >= n:
                continue

            cond_already_visited = vis[new_x][new_y] == 0

            r, g, b = data[new_x][new_y]
            min_r, min_g, min_b = colmin
            max_r, max_g, max_b = colmax

            if not (min_r <= r <= max_r and min_g <= g <= max_g and min_b <= b <= max_b):
                continue

            if cond_already_visited:
                obj = np.concatenate((obj, np.array([[new_x, new_y]], dtype=np.int32)), axis=0)
                visited = np.concatenate((visited, np.array([[new_x, new_y]], dtype=np.int32)), axis=0)
                vis[new_x][new_y] = 1

    return visited, vis


# @jit(nopython=True, cache=True)

@njit(cache=True)
def optimized_fill(obj, x, y, visited, vis, colmax, colmin, data, n, m):

    def inside(x, y, colmax, colmin, data, n, m, vis):
        new_x, new_y = x, y
        if new_x < 0 or new_y < 0 or new_x >= m or new_y >= n:
            return False

        cond_already_visited = vis[new_y][new_x] == 0

        r, g, b = data[new_y][new_x]
        min_r, min_g, min_b = colmin
        max_r, max_g, max_b = colmax

        if not (min_r <= r <= max_r and min_g <= g <= max_g and min_b <= b <= max_b):
            return False

        if cond_already_visited:
            return True
        else:
            return False
        pass

    # COMMENCEMENT ALGORITHME
    if not inside(x, y, colmax, colmin, data, n, m, vis):
        return visited, vis
    # On commence la on peut plus remplir pour regler le soucis
    """
    while inside(x, y, colmax, colmin, data, n, m, vis):
        x = x - 1
    x = x+1
    """

    obj = np.concatenate((
        obj,
        np.array([[x, x, y, 1]], dtype=np.int32)),
        axis=0)

    if y - 1 > 0:
        obj = np.concatenate((
            obj,
            np.array([[x, x, y - 1, -1]], dtype=np.int32)),
            axis=0)

    while obj.shape[0] > 0:
        coord = obj[0]
        x1, x2, y, dy = coord[0], coord[1], coord[2], coord[3]
        x = x1
        obj = obj[1:]

        if inside(x, y, colmax, colmin, data, n, m, vis):
            while inside(x - 1, y, colmax, colmin, data, n, m, vis):
                # set_pixel(x - 1, y, visited, vis, data, n, m)
                # visited = np.concatenate((visited, np.array([[x1-1, y]], dtype=np.int32)), axis=0)
                visited = np.concatenate((visited, np.array([[y, x-1]], dtype=np.int32)), axis=0)
                vis[y, x-1] = 1

                x = x - 1
            if x < x1:
                obj = np.concatenate((
                    obj,
                    np.array([[x, x1 - 1, y - dy, -dy]], dtype=np.int32)),
                    axis=0)

        while x1 <= x2:
            while inside(x1, y, colmax, colmin, data, n, m, vis):
                # set_pixel(x1, y, visited, vis, data, n, m)
                # visited = np.concatenate((visited, np.array([[x1, y]], dtype=np.int32)), axis=0)
                visited = np.concatenate((visited, np.array([[y, x1]], dtype=np.int32)), axis=0)
                vis[y, x1] = 1

                x1 = x1 + 1
            if x1 > x:
                obj = np.concatenate((
                    obj,
                    np.array([[x, x1 - 1, y + dy, dy]], dtype=np.int32)),
                    axis=0)
            if x1 - 1 > x2:
                obj = np.concatenate((
                    obj,
                    np.array([[x2 + 1, x1 - 1, y - dy, -dy]], dtype=np.int32)),
                    axis=0)
            x1 = x1 + 1
            while x1 < x2 and not inside(x1, y, colmax, colmin, data, n, m, vis):
                x1 = x1 + 1
            x = x1
    return visited, vis

# TEST AVEC EDGES & VERTICES
# TEST AVEC EDGES & VERTICES
# TEST AVEC EDGES & VERTICES


def color_diff(color1, color2):
    """
    Uses 1-norm distance to calculate difference between two values.
    """
    if isinstance(color2, tuple):
        return sum(abs(color1[i] - color2[i]) for i in range(0, len(color2)))
    else:
        return abs(color1 - color2)


# modified and took from PILLOW open source library
def flood_fill_pil_inspiration(image, xy, value, visited, vis, border=None, thresh=0):
    """
    (experimental) Fills a bounded region with a given color.

    :param image: Target image.
    :param xy: Seed position (a 2-item coordinate tuple). See
        :ref:`coordinate-system`.
    :param value: Fill color.
    :param border: Optional border value.  If given, the region consists of
        pixels with a color different from the border color.  If not given,
        the region consists of pixels having the same color as the seed
        pixel.
    :param thresh: Optional threshold value which specifies a maximum
        tolerable difference of a pixel value from the 'background' in
        order for it to be replaced. Useful for filling regions of
        non-homogeneous, but similar, colors.
    """
    pixel = np.copy(image)
    x, y = xy

    background = tuple(pixel[x, y])
    edge = {(x, y)}
    # use a set to keep record of current and previous edge pixels
    # to reduce memory consumption
    full_edge = set()
    while edge:
        new_edge = set()
        for x, y in edge:  # 4 adjacent method
            for s, t in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
                # If already processed, or if a coordinate is negative, skip
                if (s, t) in full_edge or s < 0 or t < 0:
                    continue
                try:
                    p = pixel[s, t]
                except (ValueError, IndexError):
                    pass
                else:
                    full_edge.add((s, t))
                    if border is None:
                        fill = color_diff(p, background) <= thresh
                    else:
                        fill = p != value and p != border
                    if fill:
                        # pixel[s, t] = value
                        new_edge.add((s, t))
                        # print("TEST S AND T")
                        # print(s)
                        # print(t)
                        vis[s][t] = 1
                        visited = np.concatenate((visited, np.array([[s, t]], dtype=np.int32)), axis=0)

        full_edge = edge  # discard pixels processed
        edge = new_edge
    return visited, vis

"""
@cuda.jit
def cuda_bfs_parallel(obj, visited, vis, colmax, colmin, data, n, m, moves, x, y):
    temp_obj = cuda.local.array((1024, 2), dtype=np.int64)

    while True:
        # done = True
        x, y = x,y
        temp_idx = 0

        for pos in moves:
            new_x, new_y = x + pos[0], y + pos[1]
            if new_x < 0 or new_y < 0 or new_x >= m or new_y >= n:
                continue

            cond_already_visited = vis[new_x][new_y] == 0

            r, g, b = data[new_x][new_y]
            min_r, min_g, min_b = colmin
            max_r, max_g, max_b = colmax

            cond_color = (
                (min_r <= r <= max_r) and
                (min_g <= g <= max_g) and
                (min_b <= b <= max_b)
            )

            if not cond_color:
                continue

            if cond_already_visited:
                temp_obj[temp_idx, 0] = new_x
                temp_obj[temp_idx, 1] = new_y
                vis[new_x][new_y] = 1
                visited[temp_idx, 0] = new_x
                visited[temp_idx, 1] = new_y
                done = False
                temp_idx += 1

        for j in range(temp_idx):
            obj[obj.shape[0] + j, 0] = temp_obj[j, 0]
            obj[obj.shape[0] + j, 1] = temp_obj[j, 1]

        # Réduire les threads actifs à ceux qui ont du travail restant
        active_threads = cuda.syncthreads_count(1 if not done else 0)
        if active_threads == 0:
            break
"""
