import numpy as np
from cv2 import imshow
import cuda_optimisation as gpu_optimisation
import time

ROOT_INFOBOX_TKINTER = None
SCREEN_X_FAV = 30
SCREEN_Y_FAV = 30


# recuperer les coordonées des pixels et donnes infos (aire, barycentre, min, max, ratio y/x)


def info_from_surface(pixels_list, ref_density):
    INT_INFITE = 10**26

    pixels_list = np.array(pixels_list)
    count = pixels_list.shape[0]

    c_x, c_y = 0, 0
    x_min, y_min = INT_INFITE, INT_INFITE
    x_max, y_max = 0, 0

    for i in range(count):
        y, x = pixels_list[i][0], pixels_list[i][1]
        c_x += x
        c_y += y
        if x < x_min:
            x_min = x
        if x > x_max:
            x_max = x
        if y < y_min:
            y_min = y
        if y > y_max:
            y_max = y
    try:
        area = str(round(ref_density*count/100, 2))+" cm²"
        barycentre = "("+str(round(c_x/count, 2))+";"+str(round(c_y/count, 2))+")"
        min_coord = "("+str(x_min)+";"+str(y_min)+")"
        max_coord = "("+str(x_max)+";"+str(y_max)+")"
        ratio = str(round((y_max-y_min)/(x_max-x_min), 2))
        return [("Area", area),
                ("Barycentre", barycentre),
                ("Min x/y", min_coord),
                ("Max x/y", max_coord),
                ("ratio Height/Width", ratio)
                ]
    except Exception as e:
        print("\n-----"*2)
        print("Error", e)
        print("-----\n"*2)
        return [("Area", "None"),
                ("Barycentre", "None"),
                ("Min x/y", "None"),
                ("Max x/y", "None"),
                ("ratio Height/Width", "None")
                ]


def set_past_postion():
    global ROOT_INFOBOX_TKINTER, SCREEN_X_FAV, SCREEN_Y_FAV
    SCREEN_X_FAV, SCREEN_Y_FAV = ROOT_INFOBOX_TKINTER.winfo_x(), ROOT_INFOBOX_TKINTER.winfo_y()


def on_closing():
    global ROOT_INFOBOX_TKINTER
    print("quit application")
    set_past_postion()
    ROOT_INFOBOX_TKINTER.destroy()
    ROOT_INFOBOX_TKINTER = None

# Verifie les coordonnes pour pas depasser


def valid_coord(x, y, n, m):
    if x < 0 or y < 0:
        return False
    if y >= n or x >= m:
        return False
    else:
        return True


def colour_in_range(rmin, colour, rmax): return np.all(rmin <= colour) and np.all(colour <= rmax)


def showfounded_area(visited, n, m):
    """
    Affiche la zone visité du contour de la géométrie en blanc et tout le reste en noir
    :param visited: liste des points visités (qui sont compris dans le contour
    :param n: dimension en pixel pour x de l'image
    :param m: dimension en pixel pour y de l'image
    :return:
    """
    b = np.array([255, 255, 255])
    res = np.array([[b * visited[i][j] for j in range(n)] for i in range(m)])
    img_search = res.astype(np.uint8)
    imshow('Area found', img_search)


def draw_foundedarea(image_array, pixels_list, vis, is_using_cuda, show_traited_image):
    # return gpu_optimisation.change_color(image_array, pixels_list, vis, is_using_cuda, show_traited_image)
    return gpu_optimisation.jit_change_color(image_array, pixels_list, vis, is_using_cuda, show_traited_image)


def surface_area(x, y, range_val, image_array, showing_result, is_using_cuda):
    """
    Algorithme en BFS qui utilise la méthode "fill paint"
    :param x: positions horizontal du clic de la souris
    :param y: positions vertical du clic de la souris
    :param range_val: ecart de couleur étant accepté comme compris dans la zone de contour
    :param image_array: table 2D des pixels
    :param showing_result: permet de debug et verifier le resultat
    :param is_using_cuda: permet de choisir l'optimisation gpu avec cuda
    :return: la liste des pixels visités
    """
    n = image_array.shape[1]
    m = image_array.shape[0]
    visited = []
    vis = [[0 for _ in range(n)] for _ in range(m)]
    visited.append([y, x])
    data = np.copy(image_array)

    precolor = data[y][x]
    colmin = precolor - np.array([range_val, range_val, range_val])
    colmax = precolor + np.array([range_val, range_val, range_val])
    colmax = (255, 255, 255) if np.any(colmax >= (255, 255, 255)) else colmax
    obj = [[y, x]]

    if not is_using_cuda:
        start_time = time.time()

        while len(obj) > 0:
            # On recuper la nouvelle position pour notre BFS
            coord = obj[0]
            x = coord[0]
            y = coord[1]
            # Ensuite on sort de la file
            obj.pop(0)

            move = [
                [0, 1], [1, 0],
                [-1, 0], [0, -1]
                    ]
            for pos in move:
                cond_bound = valid_coord(x + pos[0], y+pos[1], n, m)
                if not cond_bound:
                    continue
                cond_already_visited = vis[x + pos[0]][y + pos[1]] == 0
                cond_colour = colour_in_range(colmin, data[x + pos[0]][y + pos[1]], colmax)
                if cond_already_visited and cond_colour:
                    obj.append([x + pos[0], y + pos[1]])
                    visited.append([x + pos[0], y + pos[1]])
                    vis[x + pos[0]][y + pos[1]] = 1

        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Elapsed time OPTIMISATION==FALSE: {elapsed_time} seconds")
    else:
       # obj_array = np.empty((0, 4), dtype=np.int32)
       # visited_array = np.array(visited).reshape((len(visited), 2))
       # vis_array = np.array(vis).reshape(m, n)
       # edge = np.array([y, x], dtype=np.int32).reshape((1, 2))
       # full_edge = np.empty(shape=(0, 2), dtype=np.int32)
        start_time = time.time()

        # LE plus rapide
        visited, vis = gpu_optimisation.flood_fill_optimisation_final(
            image_array,
            (y, x),
            (0, 0, 0),
            visited,
            vis,
            border=None,
            thresh=range_val
        )

        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Elapsed time OPTIMISATION==TRUE: {elapsed_time} seconds")

    # On affiche l'aire qui a été trouvé en remappant les pixsels parcourus
    # Dans les 2 cas on retournes les pixels
    try:
        if showing_result:
            showfounded_area(vis, n, m)
        return visited, vis

    except Exception:
        print("il a y une erreur")
        return visited, vis
