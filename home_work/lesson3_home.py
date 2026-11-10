import cv2
import numpy as np


# Завдання 1
# Відкрийте зображення data\lesson2\darken.png. Проведіть з
# ним наступні операції, переведіть його в HSV формат та
# обробіть канал Value наступними способами:
#  застосуйте вирівнювання гістограм
#  збільшіть значення десь на 20-50%, оскільки тут
# результат буде типу float32 та явно вийде за межі [0-255]
# застосуйте np.clip(value, 0, 255) та value.astype(np.uint8)
# Виведіть результати обох обробок на екран

def start_task1():
    img = cv2.imread('../data/lesson2/darken.png')

    cv2.imshow('origin', img)

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    h, s, v = cv2.split(hsv)
    new_v = cv2.equalizeHist(v)
    new_hsv = cv2.merge((h, s, new_v))

    cv2.imshow('new hsv', new_hsv)

    h = h.astype(np.float32) * 1.2
    s = s.astype(np.float32) * 1.35
    v = v.astype(np.float32) * 1.5

    h = np.clip(h, 0, 255)
    s = np.clip(s, 0, 255)
    v = np.clip(v, 0, 255)

    h = h.astype(np.uint8)
    s = s.astype(np.uint8)
    v = v.astype(np.uint8)

    new_hsv2 = cv2.merge((h, s, v))
    cv2.imshow('new hsv2', new_hsv2)
    cv2.waitKey(0)

start_task1()

