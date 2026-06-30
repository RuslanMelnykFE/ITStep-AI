import cv2
import numpy as np

# Завдання 1
# Відкрийте зображення data/Lenna.png. Прочитайте маски
# data/mask1.png та data/mask2.png.
# Об’єднайте дві маски в одну, скористайтесь cv2.bitwise_or()
# та виведіть результат
# Виведіть ту частину зображення, яка відповідає:
#  mask1
#  mask2
#  mask1 і mask2
# Усі пікселі які не відповідають маскам замінити на 0, перед
# застосуванням змініть тип даних у масці на bool

def start_task1():
    img = cv2.imread(
        "./data/lesson1/Lenna.png",
        cv2.IMREAD_GRAYSCALE
    )

    mask1 = cv2.imread(
        "./data/lesson1/mask1.png",
        cv2.IMREAD_GRAYSCALE
    )

    mask2 = cv2.imread(
        "./data/lesson1/mask2.png",
        cv2.IMREAD_GRAYSCALE
    )

    new_mask = cv2.bitwise_or(mask1, mask2)

    cv2.imshow('new mask', new_mask)

    mask1_bool = mask1.astype(bool)
    mask2_bool = mask2.astype(bool)
    mask_bool = mask1_bool | mask2_bool

    new_img = np.copy(img)
    new_img[~mask_bool] = 0

    cv2.imshow('original', img)
    cv2.imshow('with new mask', new_img)


    cv2.waitKey(0)


# start_task1()


# Завдання 2
# Домашнє завдання
# Виведіть зображення. Підберіть самостійно межі

def start_task2():
    img = cv2.imread(
        "./data/lesson1/baboo.jpg",
        cv2.IMREAD_GRAYSCALE
    )

    area = img[8: 52, 55: 200]

    cv2.imshow('original', img)
    cv2.imshow('area', area)
    cv2.waitKey(0)


start_task2()
