import cv2
import numpy as np




## pour changer resolution image
def rescale_image(image,xres,yres):
    scale_percent = 60 # percent of original size
    width = int(image.shape[1] * scale_percent / xres)
    height = int(image.shape[0] * scale_percent / yres)
    dim = (width, height)
    image=cv2.resize(image,dim,interpolation=cv2.INTER_AREA)
    return image




#Verifie les coordonnes pour pas depasser
def validCoord(x, y, n, m):
    if x < 0 or y < 0:
        return False
    if y >= n or x >= m:
        return False
    return True


def colourInRange(rmin,colour,rmax):
    return np.all(rmin<= colour) and np.all(colour <=rmax)


def surfaceArea(x,y,range_val):
    global IMAGE_ARRAY

    n=IMAGE_ARRAY.shape[1]-1
    m=IMAGE_ARRAY.shape[0]-1
    print(n,m)
    visited=[]
    vis=[ [0 for j in range(n)] for i in range(m) ]
    print(n,m,x,y)
    vis[y][x] = 1

    data=np.copy(IMAGE_ARRAY)

    preColor=data[y][x]
    colMin=preColor-np.array([range_val,range_val,range_val])
    colMax=preColor+np.array([range_val,range_val,range_val])

    obj=[]
    obj.append([y, x])

    while(len(obj)>0):
        #print(obj)
        #On recuper la nouvelle position pour notre BFS
        coord = obj[0]
        x = coord[0]
        y = coord[1]
        #Ensuite on sort de la file
        obj.pop(0)

        # For Upside Pixel or Cell
        if validCoord(x + 1, y, n, m) and vis[x + 1][y] == 0 and colourInRange(colMin,data[x + 1][y],colMax):
            #print(data[x+1][y]==preColor)
            obj.append([x + 1, y])
            visited.append([x + 1, y])
            vis[x + 1][y] = 1
        # For Downside Pixel or Cell
        if validCoord(x - 1, y, n, m) and vis[x - 1][y] == 0 and colourInRange(colMin,data[x - 1][y],colMax):
            obj.append([x - 1, y])
            visited.append([x - 1, y])
            vis[x - 1][y] = 1
        # For Right side Pixel or Cell
        if validCoord(x, y + 1, n, m) and vis[x][y + 1] == 0 and colourInRange(colMin,data[x][y + 1],colMax):
            obj.append([x, y + 1])
            visited.append([x, y + 1])
            vis[x][y + 1] = 1
        # For Left side Pixel or Cell
        if validCoord(x, y - 1, n, m) and vis[x][y - 1] == 0 and colourInRange(colMin,data[x][y - 1],colMax):
            obj.append([x, y - 1])
            visited.append([x, y - 1])
            vis[x][y - 1] = 1
    return len(visited)

# Create a callback function to handle mouse events
def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        print(f"Mouse clicked at position: ({x}, {y})")
        print(IMAGE_ARRAY[y][x])
        print("l'aire est de :",surfaceArea(x,y,50))




### Debut code creation IMAGE
image = cv2.imread('a4_5millimeter.jpg')
IMAGE_ARRAY = np.array(image)
BOOL_ARRAY=np.empty(IMAGE_ARRAY.shape)

gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#cv2.imshow('GrayImage', gray_image)
cv2.imshow('Image', image)
#cv2.setMouseCallback('GrayImage', mouse_callback)
cv2.setMouseCallback('Image', mouse_callback)

cv2.waitKey(0)
cv2.destroyAllWindows()

