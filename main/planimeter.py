import tkinter as tk
import numpy as np
import cv2


ROOT_INFOBOX_TKINTER = None
SCREEN_X_FAV = 30
SCREEN_Y_FAV = 30


# recuperer les coordonées des pixels et donnes infos (aire, barycentre, min, max, ratio y/x)


def info_from_surface(pixels_list, ref_density):
    INT_INFITE = 10**26
    count = len(pixels_list)
    c_x, c_y = 0, 0
    x_min, y_min = INT_INFITE, INT_INFITE
    x_max, y_max = 0, 0

    for i in range(count):
        x, y = pixels_list[i][0], pixels_list[i][1]
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

    area = str(round(ref_density*count/100, 2))+" cm²"
    barycentre = "("+str(round(c_x/count, 2))+";"+str(round(c_y/count, 2))+")"
    min_coord = "("+str(x_min)+";"+str(y_min)+")"
    max_coord = "("+str(x_max)+";"+str(y_max)+")"
    ratio = str(round((y_max-y_min)/(x_max-x_min), 2))
    return [("Aire", area),
            ("Barycentre", barycentre),
            ("Min x/y", min_coord),
            ("Max x/y", max_coord),
            ("ratio Hauteur/Largeur", ratio)
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


# display_surface_info : affiche les infos sur la geometrie


def display_surface_info(characteristics):
    global ROOT_INFOBOX_TKINTER, SCREEN_X_FAV, SCREEN_Y_FAV
    if ROOT_INFOBOX_TKINTER is not None:
        set_past_postion()
        ROOT_INFOBOX_TKINTER.destroy()
        ROOT_INFOBOX_TKINTER = None
    root = tk.Tk()
    root.title("Informations sur la surface")
    root.protocol("WM_DELETE_WINDOW", on_closing)
    frame = tk.Frame(root, padx=20, pady=10, name="infobox")

    def create_label(text):
        return tk.Label(frame, text=text, font=('Arial', 12), padx=10, pady=5, anchor='w')

    for i, (char_name, char_value) in enumerate(characteristics):
        label = create_label(f"{char_name}: {char_value}")
        label.grid(row=i, column=0, sticky='w')

    frame.grid(padx=20, pady=20)
    ROOT_INFOBOX_TKINTER = root
    root.geometry("{0}x{1}+{2}+{3}".format(300, 200, SCREEN_X_FAV, SCREEN_Y_FAV))
    root.mainloop()

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


def surface_area(x, y, range_val, image_array, showing_result):
    """
    Algorithme en BFS qui utilise la méthode "fill paint"
    :param x: positions horizontal du clic de la souris
    :param y: positions vertical du clic de la souris
    :param range_val: ecart de couleur étant accepté comme compris dans la zone de contour
    :param image_array: table 2D des pixels
    :param showing_result: permet de debug et verifier le resultat
    :return: la liste des pixels visités
    """
    n = image_array.shape[1] - 1
    m = image_array.shape[0] - 1
    # print(n, m)
    visited = []
    vis = [[0 for _ in range(n)] for _ in range(m)]
    # print(n, m, x, y)
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
        if showing_result:
            showfounded_area(vis, n, m)
        return visited
    # Je dois utiliser "except Exception" sinon si je fais seulement "except:"
    # Il va aussi prendre en compte "SystemExit" and "KeyboardInterrupt"
    except Exception:
        print("il a y une erreur")
        return visited


# Montrer les images primitives


def show_primitive():
    return None
