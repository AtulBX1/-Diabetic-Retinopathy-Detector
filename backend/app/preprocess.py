import cv2
import numpy as np


def crop_black(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
    points = cv2.findNonZero(thresh)
    if points is None:
        return img
    x, y, w, h = cv2.boundingRect(points)
    return img[y:y+h, x:x+w]


def crop_resize(img, size=224):
    img = crop_black(img)
    return cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)


def ben_graham(img, sigmaX=10):
    return cv2.addWeighted(
        img,
        4,
        cv2.GaussianBlur(img, (0, 0), sigmaX),
        -4,
        128,
    )


def preprocess(img, img_size=224):
    img = crop_resize(img, img_size)
    img = ben_graham(img)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return np.expand_dims(img.astype(np.float32), axis=0)
