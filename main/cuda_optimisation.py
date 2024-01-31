# from numba import jit
from numba import njit
import numpy as np
import cv2


@njit
def change_color_jit(image_array, vis):
    image = np.copy(image_array)
    height, width = image.shape[0], image.shape[1]
    for i in range(height):
        for j in range(width):
            if vis[i, j] == 1:
                image[i, j] = (0, 0 , 0)
    return image


def jit_change_color(image_array, coordinates, vis, is_using_optimization, show_traited_image):
    vis = np.array(vis)
    image = change_color_jit(image_array, vis)
    if show_traited_image:
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        equ = cv2.equalizeHist(gray_image)
        return cv2.cvtColor(equ, cv2.COLOR_GRAY2BGR)
    else:
        if image.shape[0] > 600 or image.shape[1] > 600:
            resize_img = cv2.resize(image, (0, 0), fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
            return cv2.cvtColor(resize_img, cv2.COLOR_RGB2BGR)  # Convert image back to BGR for OpenCV
        else:
            return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # Convert image back to BGR for OpenCV


# @jit --> permet d'accelerer le code dans le CPU
# a utiliser quand on a pas idée du nombre de thread à distribuer (fonction avec tableau qui change)


@njit(cache=True)
def bfs_jit_parallell(obj, visited, vis, colmax, colmin, data, n, m):
    moves = np.array([
        [0, 1], [1, 0],
        [-1, 0], [0, -1]
    ])

    while obj.shape[0] > 0:
        coord = obj[0]
        y, x = coord[0], coord[1]

        obj = obj[1:]  # Retirer le premier élément de obj

        for pos in moves:
            new_y, new_x = y + pos[0], x + pos[1]
            if new_x < 0 or new_y < 0 or new_x-1 >= n or new_y-1 >= m:
                continue

            cond_already_visited = vis[new_y][new_x] == 0

            r, g, b = data[new_y][new_x]
            min_r, min_g, min_b = colmin
            max_r, max_g, max_b = colmax

            if not (min_r <= r <= max_r and min_g <= g <= max_g and min_b <= b <= max_b):
                continue

            if cond_already_visited:
                obj = np.concatenate((obj, np.array([[new_y, new_x]], dtype=np.int32)), axis=0)
                visited = np.concatenate((visited, np.array([[new_y, new_x]], dtype=np.int32)), axis=0)
                vis[new_y][new_x] = 1

    return visited, vis


def flood_fill_optimisation_final(image, xy, visited, vis, colmin, colmax):
    pixel = np.array(image)
    x, y = xy
    edge = {(x, y)}
    full_edge = set()
    while edge:
        new_edge = set()
        for x, y in edge:
            for s, t in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
                if (s, t) in full_edge or s < 0 or t < 0:
                    continue
                try:
                    p = pixel[s, t]
                except (ValueError, IndexError):
                    pass
                else:
                    full_edge.add((s, t))

                    fill = colmin[0] <= p[0] <= colmax[0] and \
                           colmin[1] <= p[1] <= colmax[1] and \
                           colmin[2] <= p[2] <= colmax[2]

                    if fill:
                        new_edge.add((s, t))
                        vis[s][t] = 1
                        visited.append([s, t])
        full_edge = edge
        edge = new_edge
    return np.array(visited), vis


"""
@jit(nopython=True, cache=True, fastmath=True)
def flood_fill_opti_jit(image, xy, value, visited, vis, edge, full_edge, border=None, thresh=0):
    neighbors = np.array([[1, 0], [-1, 0], [0, 1], [0, -1]], dtype=np.int32)
    pixel = image
    x, y = xy
    background = pixel[x, y]
    while edge.shape[0] > 0:
        new_edge = np.empty(shape=(0, 2), dtype=np.int32)
        for idx in range(edge.shape[0]):
            x, y = edge[idx]
            for i in range(neighbors.shape[0]):
                s, t = x+neighbors[i, 0], y+neighbors[i, 1]
                # On verifie si deja dans la liste pour pas perdre de temps
                i = 0
                found = False
                while i < full_edge.shape[0]:
                    if full_edge[i, 0] == s and full_edge[i, 1] == t:
                        found = True
                        break
                    i += 1
                # On verifie si deja dans la liste pour pas perdre de temps
                if s < 0 or t < 0 or found:#check_in_list(full_edge, [s, t]):
                    continue
                else:
                    p = pixel[s, t]
                    full_edge = np.concatenate((full_edge, np.array([[s, t]], dtype=np.int32)), axis=0)
                    if border is None:
                        fill =  abs(p[0] - background[0]) <= thresh and \
                                abs(p[1] - background[1]) <= thresh and \
                                abs(p[2] - background[2]) <= thresh
                    else:
                        fill = (
                                (p[0] != value[0] or p[1] != value[1] or p[2] != value[2]) and
                                (p[0] != border[0] or p[1] != border[1] or p[2] != border[2])
                        )
                    if fill:
                        pixel[s, t] = value
                        new_edge = np.concatenate((new_edge, np.array([[s, t]], dtype=np.int32)), axis=0)
                        vis[s][t] = 1
                        visited = np.concatenate((visited, np.array([[s, t]], dtype=np.int32)), axis=0)

        full_edge = edge  # discard pixels processed
        edge = new_edge
    return visited, vis
"""
