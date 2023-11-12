import numpy as np
import cv2


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
    # MONTRE LA FIGURE A LA FIN ET DONNER NOMBRE DE PIXEL TROUVER
    # print(vis)
    b = np.array([255, 255, 255])
    res = np.array([[b * visited[i][j] for j in range(n)] for i in range(m)])
    img_search = res.astype(np.uint8)
    cv2.imshow('Area found', img_search)


def surface_area(x, y, range_val, image_array):
    """
    Algorithme en BFS qui utilise la méthode "fill paint"
    :param x: positions horizontal du clic de la souris
    :param y: positions vertical du clic de la souris
    :param range_val: ecart de couleur étant accepté comme compris dans la zone de contour
    :param image_array: table 2D des pixels
    :return:
    """
    n = image_array.shape[1] - 1
    m = image_array.shape[0] - 1
    print(n, m)
    visited = []
    vis = [[0 for _ in range(n)] for _ in range(m)]
    print(n, m, x, y)
    vis[y][x] = 1

    data = np.copy(image_array)

    precolor = data[y][x]
    colmin = precolor - np.array([range_val, range_val, range_val])
    colmax = precolor + np.array([range_val, range_val, range_val])
    obj = [[y, x]]

    while len(obj) > 0:
        # On recuper la nouvelle position pour notre BFS
        coord = obj[0]
        x = coord[0]
        y = coord[1]
        # Ensuite on sort de la file
        obj.pop(0)
        # Pixel à Haut
        if valid_coord(x + 1, y, n, m) and vis[x + 1][y] == 0 and colour_in_range(colmin, data[x + 1][y], colmax):
            # print(data[x+1][y]==precolor)
            obj.append([x + 1, y])
            visited.append([x + 1, y])
            vis[x + 1][y] = 1
        # Pixel à bas
        if valid_coord(x - 1, y, n, m) and vis[x - 1][y] == 0 and colour_in_range(colmin, data[x - 1][y], colmax):
            obj.append([x - 1, y])
            visited.append([x - 1, y])
            vis[x - 1][y] = 1
        # Pixel à droite
        if valid_coord(x, y + 1, n, m) and vis[x][y + 1] == 0 and colour_in_range(colmin, data[x][y + 1], colmax):
            obj.append([x, y + 1])
            visited.append([x, y + 1])
            vis[x][y + 1] = 1
        # Pixel à gauche
        if valid_coord(x, y - 1, n, m) and vis[x][y - 1] == 0 and colour_in_range(colmin, data[x][y - 1], colmax):
            obj.append([x, y - 1])
            visited.append([x, y - 1])
            vis[x][y - 1] = 1

    # On affiche l'aire qui a été trouvé en remappant les pixsels parcourus
    # Dans les 2 cas on retournes les pixels
    try:
        showfounded_area(vis, n, m)
        return len(visited)
    # Je dois utiliser "except Exception" sinon si je fais seulement "except:"
    # Il va aussi prendre en compte "SystemExit" and "KeyboardInterrupt"
    except Exception:
        print("il a y une erreur")
        return len(visited)


# Montrer les images primitives


def show_primitive():
    return None