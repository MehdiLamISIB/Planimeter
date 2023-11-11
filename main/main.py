import cv2
import numpy as np

"""
Projet :
Cela semble être un projet intéressant et un peu complexe, mais nous pouvons le diviser en étapes plus petites. 
Voici une approche générale que vous pouvez suivre pour créer votre planimètre en Python :

1. Chargement de l'image :
   FAIT - Utilisez une bibliothèque comme OpenCV ou PIL pour charger l'image.
   FAIT - Affichez l'image dans une fenêtre pour que l'utilisateur puisse sélectionner l'étalon.

2. Traitement de l'image pour délimiter les contours :
   - Appliquez des filtres d'amélioration d'image pour améliorer la qualité de l'image,
        compte tenu des imperfections dues au scanner.
   - Utilisez des techniques de détection de contours (par exemple, Canny dans OpenCV) 
        pour extraire les contours de la géométrie.

3. Sélection de l'étalon :
   - Permettez à l'utilisateur de cliquer sur les deux points de l'étalon (1 cm).
   - Calculez la distance en pixels entre ces points.
   - Enregistrez cette valeur pour la conversion ultérieure de pixels en cm².

4. Conversion des pixels en cm² :
   - Lorsque l'utilisateur clique sur la géométrie, mesurez la distance entre les points en pixels.
   - Utilisez la relation entre la distance en pixels et la distance en cm pour convertir l'aire en cm².

5. Calcul de l'aire :
   - Lorsque l'utilisateur clique sur une géométrie, 
        mesurez la surface à l'aide des coordonnées de la géométrie et de la distance en pixels.

6. Affichage de l'aire :
   - Affichez l'aire calculée dans la console ou sur l'interface graphique.
"""


def dpi_to_cm():
    pass

# pour changer resolution image


def rescale_image(image, xres, yres):
    scale_percent = 60  # percent of original size
    width = int(image.shape[1] * scale_percent / xres)
    height = int(image.shape[0] * scale_percent / yres)
    dim = (width, height)
    image = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
    return image


# Montrer les images primitives


def show_primitive():
    pass


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
    # MONTRE LA FIGURE A LA FIN ET DONNER NOMBRE DE PIXEL TROUVER
    # print(vis)
    b = np.array([255, 255, 255])
    res = np.array([[b * visited[i][j] for j in range(n)] for i in range(m)])
    img_search = res.astype(np.uint8)
    cv2.imshow('Area found', img_search)


# Algorithme en BFS


def surface_area(x, y, range_val):
    global IMAGE_ARRAY
    n = IMAGE_ARRAY.shape[1] - 1
    m = IMAGE_ARRAY.shape[0] - 1
    print(n, m)
    visited = []
    vis = [[0 for _ in range(n)] for _ in range(m)]
    print(n, m, x, y)
    vis[y][x] = 1

    data = np.copy(IMAGE_ARRAY)

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
    except:
        print("il a y une erreur")
        return len(visited)


# EVENEMENT SOURIS


def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        print(f"Mouse clicked at position: ({x}, {y})")
        print(IMAGE_ARRAY[y][x])
        range_colorval = 20
        print("l'aire est de :", surface_area(x, y, range_colorval))


# Debut code creation IMAGE


scanner_image = cv2.imread('../scan_home/100_PPP.png')
IMAGE_ARRAY = np.array(scanner_image)
BOOL_ARRAY = np.empty(IMAGE_ARRAY.shape)
gray_image = cv2.cvtColor(scanner_image, cv2.COLOR_BGR2GRAY)
# cv2.imshow('GrayImage', gray_image)
cv2.imshow('Image', scanner_image)
# cv2.setMouseCallback('GrayImage', mouse_callback)
cv2.setMouseCallback('Image', mouse_callback)
cv2.waitKey(0)
cv2.destroyAllWindows()
