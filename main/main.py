import cv2
import numpy as np
import planimeter as planimeter
import reference as ref
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
from time import sleep

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
   CHANGER - Permettez à l'utilisateur de cliquer sur les deux points de l'étalon (1 cm).
   CHANGER - Calculez la distance en pixels entre ces points.
   FAIT - Cliquer sur la geometrie de reference et calculer le nombre de pixels
   FAIT - Enregistrez cette valeur pour la conversion ultérieure de pixels en cm².

4. Conversion des pixels en cm² :
   FAIT - Lorsque l'utilisateur clique sur la géométrie, mesurez la distance entre les points en pixels.
   FAIT - Utilisez la relation entre la distance en pixels et la distance en cm pour convertir l'aire en cm².

5. Calcul de l'aire :
   FAIT - Lorsque l'utilisateur clique sur une géométrie, 
        mesurez la surface à l'aide des coordonnées de la géométrie et de la distance en pixels.

6. Affichage de l'aire :
   FAIT (mais doit optimiser) - Affichez l'aire calculée dans la console ou sur l'interface graphique.
"""


# EVENEMENT SUIVIES

EVENT_REFERENCE_START = False
EVENT_REFERENCE_DONE = False
EVENT_PLANIMETER_MESUREMENT = False
EVENT_IMAGE_SET = False

# VALEUR GLOBALE

REF_POS = [0, 0, 0, 0]
REF_DENSITY = None

COEFF_X = 1
COEFF_Y = 1
REF_COEFF_USER = 1
MAINWINDOW_WIDTH = 600
MAINWINDOW_HEIGHT = 600

IMAGE_ARRAY = None
COLOR_RANGE = 15

WINDOW_SHOW_AREA_INFO = None
# EVENEMENT SOURIS

# Application principale


def main_application():
    global EVENT_IMAGE_SET, REF_COEFF_USER

    root = tk.Tk()
    root.title("Planimeter")
    root.geometry("600x600")
    root.resizable(False, False)

    # TKINTER NATIVE VARIABLE

    TOOGLE_CUDA_CHOOSE = tk.BooleanVar()
    TOOGLE_CUDA_CHOOSE.set(True)

    # CANVAS FUNCTION AND EVENT

    def show_caracteristic_area(characteristics):
        global WINDOW_SHOW_AREA_INFO

        def on_closing():
            global WINDOW_SHOW_AREA_INFO
            WINDOW_SHOW_AREA_INFO.destroy()
            WINDOW_SHOW_AREA_INFO = None

        def create_label(text):
            return tk.Label(WINDOW_SHOW_AREA_INFO, text=text, font=('Arial', 12), padx=10, pady=5, anchor='w')

        if WINDOW_SHOW_AREA_INFO is None:
            WINDOW_SHOW_AREA_INFO = tk.Toplevel(root)
            WINDOW_SHOW_AREA_INFO.title("Info about area ...")

            for i, (char_name, char_value) in enumerate(characteristics):
                label = create_label(f"{char_name}: {char_value}")
                label.grid(row=i, column=0, sticky='w')
            WINDOW_SHOW_AREA_INFO.geometry("300x200")
            WINDOW_SHOW_AREA_INFO.protocol("WM_DELETE_WINDOW", on_closing)
        else:
            for widgets in WINDOW_SHOW_AREA_INFO.winfo_children():
                widgets.destroy()

            for i, (char_name, char_value) in enumerate(characteristics):
                label = create_label(f"{char_name}: {char_value}")
                label.grid(row=i, column=0, sticky='w')


    def set_reference(event):
        global EVENT_IMAGE_SET, EVENT_REFERENCE_DONE, COEFF_X, COEFF_Y, REF_DENSITY, IMAGE_ARRAY, COLOR_RANGE

        x = event.x
        y = event.y

        print(f"Clicked at (x={x}, y={y})")
        if EVENT_IMAGE_SET:
            img_x_pos = int(x*COEFF_X)
            img_y_pos = int(y*COEFF_Y)
            canvas.config(cursor="watch")
            ref_visited, ref_vis = planimeter.surface_area(
                    img_x_pos,
                    img_y_pos,
                    COLOR_RANGE,
                    IMAGE_ARRAY,
                    showing_result=False,
                    is_using_cuda=False)

            REF_DENSITY = ref.mm_area_of_pixel_unit_with_counts_know(
                np.array(ref_visited).shape[0]
            )

            choose_value_reference_window()

            EVENT_REFERENCE_DONE = True
            canvas.config(cursor="crosshair")
            # print("Reference calculé")
        else:
            canvas.config(cursor="crosshair")
            print("besoin d'avoir une image !!!!!")

    def mouse_callback(event):
        global EVENT_IMAGE_SET, EVENT_REFERENCE_DONE, COEFF_X, COEFF_Y, REF_DENSITY, IMAGE_ARRAY, COLOR_RANGE
        x = event.x
        y = event.y
        # print(f"Clicked at (x={x}, y={y})")
        # print("Is using CUDA --> ", TOOGLE_CUDA_CHOOSE.get())
        # Si une image est affiché, on peut lancer l'algo
        if EVENT_IMAGE_SET:
            img_x_pos = int(x*COEFF_X)
            img_y_pos = int(y*COEFF_Y)
            # print(f"Clicked at (x={img_x_pos}, y={img_y_pos})")
            if not EVENT_REFERENCE_DONE:
                print("la reference n'as pas encore ete mis ")
                # return None
            else:
                canvas.config(cursor="watch")

                pixel_list, vis = planimeter.surface_area(img_x_pos,
                                                     img_y_pos,
                                                     COLOR_RANGE,
                                                     IMAGE_ARRAY,
                                                     showing_result = False,
                                                     is_using_cuda = TOOGLE_CUDA_CHOOSE.get())

                cv2.imshow('Area selectionned', planimeter.draw_foundedarea(IMAGE_ARRAY, pixel_list, vis, TOOGLE_CUDA_CHOOSE.get(), False))
                # je dessine d'abord car après quand la fenêtre est ouverte, l'application est focus sur cette fenêtre
                show_caracteristic_area(planimeter.info_from_surface(pixel_list, REF_DENSITY*REF_COEFF_USER))
                canvas.config(cursor="crosshair")

    # CANVAS

    canvas = tk.Canvas(root, width=600, height=600, bg="black")
    canvas.config(cursor="crosshair")
    canvas.bind('<Button-1>', mouse_callback)
    canvas.bind('<Button-3>', set_reference)
    canvas.pack()

    def credit_app():
        msg = "Application created by Mehdi Lamrabet\n Planimeter - 2024©"
        credit_window = tk.Toplevel(root)
        credit_window.title("Credit")
        credit_window.geometry("400x50")
        credit_window.wm_resizable(False, False)
        credit_window.config()
        tk.Label(credit_window, font=("Courier New", 12), text=msg).pack()

        # Positionnement a peu pres au centre
        x = root.winfo_rootx() + 100
        y = root.winfo_rooty() + 100
        credit_window.geometry(f"+{x}+{y}")
        # Focus sur la fenêtre (permet d'éviter d'avoir des duplications
        credit_window.grab_set()

    def choose_value_reference_window():
        ref_value_window = tk.Toplevel(root)

        def validate_numbers():
            global REF_COEFF_USER
            input_data = entry.get()
            if input_data:
                try:
                    float(input_data)
                    label.config(
                        text=f"Valid numeric value: {input_data}",
                        foreground="green",
                    )
                    REF_COEFF_USER = abs(float(input_data))
                    #time_python.sleep(0.2)
                    sleep(0.2)
                    ref_value_window.destroy()
                except ValueError:
                    label.config(
                        text=f'Numeric value expected, got "{input_data}"',
                        foreground="red",
                    )
            else:
                label.config(text="Entry is empty", foreground="red",)

        def validate_with_enter(event):
            validate_numbers()

        ref_value_window.title("Choose your reference")
        ref_value_window.geometry("300x100")
        ref_value_window.wm_resizable(False, False)
        ref_value_window.config()

        # Entree
        entry = tk.Entry(ref_value_window, width=35)
        entry.grid(row=0, column=0, padx=5, pady=5)
        entry.bind("<Return>", validate_with_enter)

        button = tk.Button(ref_value_window, text="Validate", command=validate_numbers)
        button.grid(row=0, column=1, padx=5, pady=5)
        label = tk.Label(ref_value_window, text="Value of reference [cm²]")
        label.grid(row=1, column=0, columnspan=2, padx=5, pady=5)

        # Positionnement a peu pres au centre
        x = root.winfo_rootx() + 100
        y = root.winfo_rooty() + 100
        ref_value_window.geometry(f"+{x}+{y}")
        # Focus sur la fenêtre (permet d'éviter d'avoir des duplications
        ref_value_window.grab_set()

    def close_app():
        root.quit()

    def open_image():
        global COEFF_X, COEFF_Y, EVENT_REFERENCE_DONE
        if EVENT_REFERENCE_DONE:
            # Permet de remttre à zéro à chaque nouvelle image
            EVENT_REFERENCE_DONE = False
        file_path = filedialog.askopenfilename(
            title="Select your picture",
            filetypes=(("Image files", "*.jpg;*.jpeg;*.png"),)
        )

        if file_path:
            file_path = file_path.replace("\\ ", " ")
            print(file_path)
            img = cv2.imread(file_path)
            height, width, channels = img.shape

            # DX_MAX = 1280
            # DY_MAX = 1200

            # DX_MAX = 1600
            # DY_MAX = 720

            DX_MAX = 1169
            DY_MAX = 827

            if width > DX_MAX and height > DY_MAX:
                coeff_x = float(DX_MAX / width)
                coeff_y = float(DY_MAX / height)
                coeff = coeff_y if coeff_y < coeff_x else coeff_x

                COEFF_X = 1 / coeff
                COEFF_Y = 1 / coeff
                # COEFF DE 0.5
                # resize_img = cv2.resize(img, (0, 0), fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
                resize_img = cv2.resize(img, (0, 0), fx=coeff, fy=coeff, interpolation=cv2.INTER_AREA)
                # cv2.imshow("Image",resize_img)
                # cv2.moveWindow("Image",500,500)
                show_image(resize_img)
            elif width > DX_MAX:
                coeff = float(DX_MAX / width)
                COEFF_X = 1 / coeff
                COEFF_Y = 1 / coeff
                resize_img = cv2.resize(img, (0, 0), fx=coeff, fy=coeff, interpolation=cv2.INTER_AREA)
                show_image(resize_img)
            elif height > DY_MAX:
                coeff = float(DY_MAX / height)
                COEFF_X = 1 / coeff
                COEFF_Y = 1 / coeff
                resize_img = cv2.resize(img, (0, 0), fx=coeff, fy=coeff, interpolation=cv2.INTER_AREA)
                show_image(resize_img)
            else:
                COEFF_X = 1
                COEFF_Y = 1
                show_image(img)

    def show_image(img):
        global EVENT_IMAGE_SET, IMAGE_ARRAY
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        IMAGE_ARRAY = rgb_img

        pil_img = Image.fromarray(rgb_img)
        imgtk = ImageTk.PhotoImage(image=pil_img)

        canvas.create_image(0, 0, image=imgtk, anchor=tk.NW)
        canvas.image = imgtk
        root.geometry(f"{img.shape[1]}x{img.shape[0]}")
        canvas.config(width=img.shape[1], height=img.shape[0])
        EVENT_IMAGE_SET = True

    # MENU

    menu = tk.Menu(root)
    root.config(menu=menu)

    # MENU - file manager
    file_menu = tk.Menu(menu, tearoff=0)
    menu.add_cascade(label="File", menu=file_menu)
    file_menu.add_command(label="Open Image", command=open_image)

    # MENU - application manager

    app_menu = tk.Menu(menu, tearoff=0)
    menu.add_cascade(label="Application", menu=app_menu)
    app_menu.add_checkbutton(label="Optimisation", variable=TOOGLE_CUDA_CHOOSE, onvalue=True, offvalue=False)
    app_menu.add_command(label="Close", command=close_app)
    app_menu.add_command(label="Credit", command=credit_app)

    root.mainloop()


if __name__ == "__main__":
    main_application()
