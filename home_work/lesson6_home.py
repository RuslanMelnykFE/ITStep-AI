import ultralytics
import cv2

# створення моделі
model = ultralytics.YOLO('yolov8s.pt')

# ========================================================================================== #
# Завдання 1
# Відкрийте відео з файлу data\lesson8\meetings.mp4
# Застосуйте детекцію та виведіть результат, підберіть параметри
# Можете змінити розмір кадру для кращої візуалізації cv2.resize()

def start_task1():
    cap = cv2.VideoCapture('../data/lesson8/meetings.mp4')

    while True:
        success, img = cap.read()
        img = cv2.resize(img, None, fx=0.5, fy=0.5)

        if not success:
            break

        results = model.predict(
            img,
            device='mps',
            conf=0.7,
            iou=0.5
        )[0]

        cv2.imshow('original', results.plot())

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# start_task1()


# ========================================================================================== #
# Завдання 2
# Відкрийте відео з файлу data\lesson8\meetings.mp4
# Застосуйте детекцію та почніть показувати відео з
# моменту, коли людей стало 5

def start_task2():
    cap = cv2.VideoCapture('../data/lesson8/meetings.mp4')
    is_found = False

    while True:
        success, img = cap.read()
        img = cv2.resize(img, None, fx=0.5, fy=0.5)
        people_counter = 0

        if not success:
            break

        results = model.predict(
            img,
            device='mps',
            conf=0.7,
            iou=0.5
        )[0]

        boxes = results.boxes

        for box in boxes:
            if int(box.cls[0] == 0):
                people_counter += 1

        if people_counter >= 5:
            is_found = True

        if is_found:
            cv2.imshow('original', results.plot())

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

start_task2()
