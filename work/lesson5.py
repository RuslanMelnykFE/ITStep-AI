import cv2


# cap = cv2.VideoCapture(0)
#
# width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
# height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
#
# print(width, height)

# fps = int(cap.get(cv2.CAP_PROP_FPS))
#
# print(fps)

# success, frame = cap.read()
#
# cv2.imshow('frame',frame)
# cv2.waitKey(0)

#===============================================================================================================#
# Завдання 1
# Виведіть відео з файлу data\lesson7\text.mp4 на екран та збережіть в новий файл.
# Змініть розмір зображення.

def start_task1():
    cap = cv2.VideoCapture('../data/lesson7/text.mp4')
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    new_width = width * 2
    new_height = height * 2

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    result = cv2.VideoWriter(
        '../data/lesson7/output.avi',
        fourcc,
        fps,
        (new_width, new_height))


    while True:
        success, frame = cap.read()
        if not success:
            break

        resized_frame = cv2.resize(
            frame,
            (new_width, new_height),
            interpolation=cv2.INTER_AREA
        )

        cv2.imshow("Відео", resized_frame)

        result.write(resized_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    result.release()
    cv2.destroyAllWindows()

# start_task1()


#===============================================================================================================#
# Завдання 2
# Відкрийте відео з файлу data\lesson7\text.mp4. Проведіть
# бінарізацію кадрів та збережіть в новий файл.

def start_task2():
    cap = cv2.VideoCapture('../data/lesson7/text.mp4')
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    result = cv2.VideoWriter(
        '../data/lesson7/output_binary.avi',
        fourcc,
        fps,
        (width, height))


    while True:
        success, frame = cap.read()

        if not success:
            break

        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        frame_binary = cv2.adaptiveThreshold(
            frame_gray,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            9,
            3
        )

        cv2.imshow("Відео", frame_binary)

        result.write(frame_binary)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    result.release()
    cv2.destroyAllWindows()

# start_task2()


#===============================================================================================================#
# Завдання 3
# Відкрийте відео з файлу data\lesson7shapes.mp4.
# Проведіть виділення країв на кадрах та збережіть в новий файл.

def set_border(img, width, height):
    border = 30
    border_top = border
    border_left = int(width / 3)
    border_bottom = height - border
    border_right = int(width / 3 * 2)

    img_res = img.copy()

    img_res[:border_top] = 100
    img_res[border_bottom:] = 100
    img_res[:, :border_left] = 100
    img_res[:, border_right:] = 100

    return img_res

def get_mask_blue(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    lower_blue = (90, 100, 150)
    upper_blue = (150, 255, 255)

    return cv2.inRange(hsv, lower_blue, upper_blue)


def start_task3():
    cap = cv2.VideoCapture('../data/lesson7/shapes.mp4')
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    result = cv2.VideoWriter(
        '../data/lesson7/output_shapes.avi',
        fourcc,
        fps,
        (width, height))


    while True:
        success, frame = cap.read()

        if not success:
            break

        cv2.imshow("Відео", frame)

        mask_blue = get_mask_blue(frame)
        cv2.imshow("Відео mask", mask_blue)

        frame_border = set_border(frame, width, height)
        cv2.imshow('frame_border', frame_border)

        result.write(frame_border)

        if cv2.waitKey(30) & 0xFF == ord('q'):
            break

    cap.release()
    result.release()
    cv2.destroyAllWindows()

start_task3()