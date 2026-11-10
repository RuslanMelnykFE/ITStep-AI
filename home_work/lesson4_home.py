import cv2
from utils import trackbar_decorator

# Завдання 1
# Відкрийте зображення data/lesson3/sonet.png. Проведіть бінарізацію.
# Обов’язково використайте:
#  розмиття або наведення різкості
#  адаптивну бінарізацію
#  очищеня шумів

def start_task1():
    img = cv2.imread('../data/lesson3/sonet.png')

    cv2.imshow('original', img)

    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    img_gauss = cv2.GaussianBlur(img_gray, (3, 3), 0)

    img_binary = cv2.adaptiveThreshold(
        img_gauss,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        9,
        1
    )

    result = cv2.fastNlMeansDenoising(
        img_binary,
        None,
        h=20,
        templateWindowSize=15,
        searchWindowSize=41)

    cv2.imshow('result', result)
    cv2.waitKey(0)

# start_task1()







# Завдання 2
# Відкрийте зображення data/lesson3/sonnet_noised.png.
# Проведіть бінарізацію. Застосуйте код з завдання 1 та
# спробуйте покращити результат

def start_task2():
    img = cv2.imread('../data/lesson3/sonet_noised.png')

    cv2.imshow('original', img)

    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    img_gauss = cv2.GaussianBlur(img_gray, (3, 3), 0)

    img_binary = cv2.adaptiveThreshold(
        img_gauss,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        7,
        3
    )

    result = cv2.fastNlMeansDenoising(
        img_binary ,
        None,
        h=20,
        templateWindowSize=21,
        searchWindowSize=40)

    cv2.imshow('result', result)
    cv2.waitKey(0)

start_task2()
