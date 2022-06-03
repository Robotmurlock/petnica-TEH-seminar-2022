import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from typing import Optional


IMAGES_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'images'))


def load_image(name: str, grayscale: bool = False) -> np.ndarray:
    """
    Ucitava sliku iz images/ direktorijuma

    :param name: Ime slike u images direktorijumu
    :param grayscale: Da li slika treba da bude crno bela?
    :return: Ucitavana slika
    """
    # ucitava sliku u RGB (BGR) formatu
    img = cv2.imread(os.path.join(IMAGES_PATH, name))  
    print(os.path.join(IMAGES_PATH, name))
    assert img is not None, 'Loading image failed!'

    if grayscale:
        # konvertuje sliku iz RGB (BGR) formata
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    return img
    

def plot_image(img: np.ndarray, fig: Optional[plt.Figure] = None) -> plt.Figure:
    """
    Iscrtava sliku

    :param img: Slika
    :param fig: Figura (opciono)
    :return: Figura
    """
    if fig is None:
        fig = plt.figure(figsize=(14, 10))
    else:
        fig.clf()

    img_local = img
    if len(img.shape) > 2:
        # BGR -> RGB
        img_local = img.copy()[:, :, ::-1]

    plt.imshow(img_local)
    plt.axis('off')
    return fig


def add_axis(img: np.ndarray) -> np.ndarray:
    """
    Iscrtava koordinatni sistem na slici

    :param img: Slika
    :return: Slika
    """
    img = cv2.arrowedLine(img, [5, 5], [5, img.shape[0] // 4], (255, 0, 0), 10)
    img = cv2.arrowedLine(img, [5, 5], [img.shape[1] // 4, 5], (0, 0, 255), 10)
    img = cv2.circle(img, [5, 5], 5, (0, 255, 0), 12)
    return img


def transform(img: np.ndarray, M: Optional[np.ndarray] = None, T: Optional[np.ndarray] = None, trans_point: Optional[np.ndarray] = None) \
        -> np.ndarray:
    """
    Izvrsava transformaciju sliku

    :param img: Slika
    :param M: 2d matrica transformacije
    :param T: Vektor translacije
    :param trans_point: Tacka u odnosu na koju se primenjuje transformacija
    :return: Transformisana slika
    """

    if M is None:
        # Ako M nije definisano, onda se uzima neutral
        M = np.eye(2)

    if T is None:
        # Ako T nije definisano, onda se uzima neutral
        T = np.array([[0, 0]]).T

    transformed_img = np.zeros_like(img)  # pravimo praznu matricu istog formata kao i img
    n_rows, n_cols = img.shape[0], img.shape[1]

    for row in range(n_rows):
        for col in range(n_cols):
            point = np.array([[row, col]]).T  # izdvajaju se koordinate iz ulazne slike
            if trans_point is not None:
                # pomeranje koordinatnog sistema
                point -= trans_point

            transformed_point = M @ point + T  # transformisu se koordinate ulazne slike
            if trans_point is not None:
                # vracanje koordinatnog sistema
                transformed_point += trans_point

            # transformisane koordinate se pretvaraju u int
            transformed_row, transformed_col = int(transformed_point[0, 0]), int(transformed_point[1, 0]) 

            if 0 <= transformed_row < n_rows and 0 <= transformed_col < n_cols:
                # Ako nova koordinata izlazi iz opsega slike, onda se ignorise
                transformed_img[transformed_row, transformed_col] = img[row, col]  # postavlja se vrednost koordinate na novu sliku

    return transformed_img


def fast_transform(img: np.ndarray, M: Optional[np.ndarray] = None, T: Optional[np.ndarray] = None, trans_point: Optional[np.ndarray] = None) \
        -> np.ndarray:
    """
    Isto kao i `transform`, ali optimizovana verzija koja koristi opencv

    :param img: Slika
    :param M: 2d matrica transformacije
    :param T: Vektor translacije
    :param trans_point: Tacka u odnosu na koju se primenjuje transformacija
    :return: Transformisana slika
    """

    TP = np.array([
        [1, 0, -trans_point[0, 0]],
        [0, 1, -trans_point[1, 0]],
        [0, 0, 1]
    ], dtype=np.float32)

    TP_inv = np.array([
        [1, 0, trans_point[0, 0]],
        [0, 1, trans_point[1, 0]],
        [0, 0, 1]
    ], dtype=np.float32)
    
    H = np.array([
        [M[0, 0], M[0, 1], T[0, 0]],
        [M[1, 0], M[1, 1], T[1, 0]],
        [0, 0, 1]
    ], dtype=np.float32)

    H_point = TP_inv @ H @ TP

    return cv2.warpPerspective(img, H_point, (img.shape[1], img.shape[0]))
