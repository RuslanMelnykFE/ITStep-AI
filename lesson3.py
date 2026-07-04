import cv2
import numpy as np


# Завдання 1
# Відкрийте зображення data/lesson2/marbles.png.
# Використайте кольорову сегментацію для отримання масок до
# кульок:
#  синього кольору
#  зеленого і червоного
#  чорного
#  білого
#  усіх кульок

def start_task1():
    img = cv2.imread('data/lesson2/marbles.png')

    cv2.imshow('origin', img)

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    #  синього кольору
    lower_blue = (90, 100, 150)
    upper_blue = (150, 255, 255)

    mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)
    cv2.imshow('mask blue', mask_blue)

    #  зеленого і червоного
    lower_green = (40, 90, 80)
    upper_green = (90, 255, 255)
    mask_green = cv2.inRange(hsv, lower_green, upper_green)

    lower_red1 = (0, 110, 100)
    upper_red1 = (10, 255, 255)
    mask_red1 = cv2.inRange(hsv, lower_red1, upper_red1)

    lower_red2 = (170, 110, 100)
    upper_red2 = (180, 255, 255)
    mask_red2 = cv2.inRange(hsv, lower_red2, upper_red2)

    mask_red = cv2.bitwise_or(mask_red1, mask_red2)

    mask_green_red = cv2.bitwise_or(mask_green, mask_red)

    cv2.imshow('mask green red', mask_green_red)

    # чорного
    lower_black = (0, 0, 0)
    upper_black = (180, 90, 40)
    mask_black = cv2.inRange(hsv, lower_black, upper_black)

    cv2.imshow('mask black', mask_black)

    #  білого
    lower_white = (0, 0, 200)
    upper_white = (180, 50, 255)
    mask_white = cv2.inRange(hsv, lower_white, upper_white)
    cv2.imshow('mask white', mask_white)

    white_black = cv2.bitwise_and(mask_white, mask_black)
    wbgr = cv2.bitwise_or(white_black, mask_green_red)
    mask_all = cv2.bitwise_or(mask_blue, wbgr)
    cv2.imshow('mask all', mask_all)

    cv2.waitKey(0)

# start_task1()


# Завдання 2
# Відкрийте зображення data/lesson2/cell.png. Покращте
# зображення за допомогою вирівнювання гістограми. Оскільки
# зображення кольорове, вам доведеться зробити наступні
# кроки:
#  перевести зображення в LAB
#  розбити зображення на канали l, a та b
#  вирівняти гістограму для l
#  зібрати канали назад в зображення
#  перевести результат назад в BGR
# Порівняйте результати для 2 алгоритмів.


def start_task2():
    img = cv2.imread('data/lesson2/cell.png')

    cv2.imshow('origin', img)

    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.equalizeHist(l)
    new_lab = cv2.merge((l, a, b))
    new_img = cv2.cvtColor(new_lab, cv2.COLOR_LAB2BGR)

    cv2.imshow('new lab', new_img)

    cv2.waitKey(0)

start_task2()





# # дз
# result1 = img.copy()
# result1[:] = 0
#
# # те саме(створити масив 0 такого ж розміру як img)
# result1 = np.zero_like(img)
#
# # проблема зі знаками \
# r"C:\ITSTEP\ITStep-AI\data\lesson1\baboo.jpg"


# # колір
# import cv2
#
# # читати як чорно біле
# img = cv2.imread('data/lesson2/lego.jpg', cv2.IMREAD_GRAYSCALE)
# img = cv2.resize(img, (500, 500))
#
# print(img.shape)
# print(img.dtype)
#
# cv2.imshow('gray', img)
#
# # читати як кольорове
# img = cv2.imread('data/lesson2/lego.jpg', cv2.IMREAD_COLOR)
# img = cv2.resize(img, (500, 500))
#
# print(img.shape)
# print(img.dtype)
#
# # піксель формат bgr
# print(img[220, 220]) # [60 75 230]
#
# cv2.imshow('color', img)
#
# # дістати черіоний колір з зображення
# # img.shape = (рядки, стовпчики, колір)
# red_part = img[:, :, 2]
# green_part = img[:, :, 1]
# blue_part = img[:, :, 0]
#
# print(red_part.shape)
# print(red_part.dtype)
#
# # якщо в shape 2 числа -- показує як чорнобіле зображення
# cv2.imshow('wrong red', red_part)
#
# # правильно
# red_part = img.copy()
# # первести синій та зелений в 0
# red_part[:, :, 0] = 0  # blue
# red_part[:, :, 1] = 0  # blue
#
# cv2.imshow('red', red_part)
# cv2.waitKey(0)
#
#
# # # rgb(bgr)
# # import utils
# #
# # utils.lesson2_bgr_range()

# кольоровий простір hsv
# h -- колір  (hue)  -- кут / 2
# s -- насиченість  (saturation)
# v -- яскравість   (value)

# import utils
#
# utils.lesson2_hsv_range()

# import cv2
#
#
# img = cv2.imread('data/lesson2/lego.jpg', cv2.IMREAD_COLOR)
# img = cv2.resize(img, (500, 500))
#
# # img -- bgr
# cv2.imshow('bgr', img)
#
# # перевести в hsv
# hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
#
# print(hsv.shape)
# print(hsv.dtype)
#
# # # думає що hsv -- bgr
# # cv2.imshow('hsv', hsv)
#
# # h -- 100-130
# # s -- 150-255
# # v -- 140-255
# mask_blue = cv2.inRange(
#     hsv,
#     (100, 150, 140), # нижні межі
#     (130, 255, 255)  # верхні межі
# )
#
# print(mask_blue.shape)
# print(mask_blue.dtype)
#
# cv2.imshow('mask_blue', mask_blue)
#
# cv2.waitKey(0)


# import cv2
#
#
# img = cv2.imread('data/lesson2/bio_low_contrast.jpg', cv2.IMREAD_GRAYSCALE)
# img = cv2.resize(img, (500, 500))

# img -- bgr
# cv2.imshow('origin', img)


# наведення різкості(конраст)
# вирівнювання гістрограм(працює лише для чорнобілих)
# result = cv2.equalizeHist(img)
#
# cv2.imshow('result', result)
#
# img = cv2.imread('data/lesson2/cell.png')
# hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# вирівнюємо яскравість
# value = hsv[:, :, 2]
# new_value = cv2.equalizeHist(value)
# hsv[:, :, 2] = new_value

# перевести назад в bgr
# new_img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
#
# cv2.imshow('contrast', new_img)
# cv2.imshow('origin', img)
# cv2.waitKey(0)