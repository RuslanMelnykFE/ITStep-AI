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
    img = cv2.imread('../data/lesson2/marbles.png')

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
    img = cv2.imread('../data/lesson2/cell.png')

    cv2.imshow('origin', img)

    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.equalizeHist(l)
    new_lab = cv2.merge((l, a, b))
    new_img = cv2.cvtColor(new_lab, cv2.COLOR_LAB2BGR)

    cv2.imshow('new lab', new_img)

    cv2.waitKey(0)

start_task2()
