import cv2
import numpy as np


# Завдання 1
# Відкрийте зображення data/Lenna.png. Виведіть на екран
# розмір зображення, тип даних, максимальну та мінімальну
# інтенсивність пікселів, саме зображення з підписом.
img = cv2.imread(
        'data/lesson1/Lenna.png',  # шлях до файлу
        cv2.IMREAD_GRAYSCALE # формат зображення
    )


def start_task1():
    cv2.imshow('original', img)
    cv2.waitKey(0)
    print(f"розмір зображення: {img.shape}")
    print(f"тип даних: {img.dtype}")
    print(f"максимальну інтенсивність пікселі: {img.max()}")
    print(f"мінімальну інтенсивність пікселі: {img.min()}")


# start_task1()


# Завдання 2
# Відкрийте зображення data/Lenna.png. Виведіть на екран
# такі зображень:
#  Верхній лівий кут розміром 100х50
#  Центральний квадрат розміром 100х100
#  Верхню половину
#  Нижню половину
#  Ліву половину
#  Праву половину


def start_task2():
    top_left = img[:101, :51]
    center = img[77:178, 77:178]
    top = img[:128]
    bottom = img[128:]
    left = img[:, :128]
    right = img[:, 128:]

    cv2.imshow('original', img)
    cv2.imshow('top_left', top_left)
    cv2.imshow('center', center)
    cv2.imshow('top', top)
    cv2.imshow('bottom', bottom)
    cv2.imshow('left', left)
    cv2.imshow('right', right)
    cv2.waitKey(0)


# start_task2()


# Завдання 3
# Відкрийте зображення data/Lenna.png. Створіть наступні зображення
def start_task3():
    border = 30
    border_top = border
    border_left = border
    border_bottom = 256 - border
    border_right = 256 - border

    img2 = img.copy()
    img3 = img.copy()
    img4 = img.copy()

    img2[:border_top] = 0
    img2[border_bottom: ] = 255

    img3[:, :border_left] = 0
    img3[:, border_right:] = 0

    img4[:border_top] = 0
    img4[border_bottom:] = 0
    img4[:, :border_left] = 0
    img4[:, border_right:] = 0

    cv2.imshow('original', img)
    cv2.imshow('top_bottom', img2)
    cv2.imshow('lef_right', img3)
    cv2.imshow('all', img4)
    cv2.waitKey(0)


# start_task3()


# Завдання 4
# Відкрийте зображення data/Lenna.png. Створіть маску для
# пік селів з інтенсивністю більше 128 та виведіть її. Також
# виведіть заперечення цієї маски.
# На оригінальному зображенні, усі пікселі які не
# відповідають масці замініть на 0 та виведіть результат

def start_task4():
    mask = img > 128

    print(f"{img[mask] = }")
    print(f"{img[~mask] = }")

    new_mask = mask.astype(np.uint8) * 255


    img2 = img.copy()
    img2[~mask] = 0
    cv2.imshow('original', img)
    cv2.imshow('with new mask', new_mask)
    cv2.imshow('change origin', img2)
    cv2.waitKey(0)


start_task4()

# гамма корекція
