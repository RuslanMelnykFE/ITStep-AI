import ultralytics
import cv2
from torch.compiler.config import verbose

# створення моделі
model = ultralytics.YOLO('yolov8s.pt')


def start_practice():
    # отримати зображення з відео
    cap = cv2.VideoCapture('../data/lesson8/cars+bikes.mp4')
    success, img = cap.read()

    img = cv2.resize(img, None, fx=0.5, fy=0.5)

    cv2.imshow('original', img)

    results = model.predict(
        img,            # саме зображення
        device='mps',   # процесор де запускати модель
                        # cpu -- звичайний процес
                        # cuda -- графічний процесор
        conf=0.5,       # мінімальній відсоток для об'єкта щоб попасти в result
        iou=0.7,        # максимально можливий рівень перетину рамок(якщо більше то вважаємо
                        # що це рамки для одного й того самого об'єкта)
        classes=[0, 1]  # id класів які детектити
    )

    results = results[0]

    # names = results.names

    # print(names)

    boxes = results.boxes

    # print(f'boxes: \n {boxes}')

    conf = boxes.conf
    conf = conf.cpu().numpy()
    print(conf)

    cv2.waitKey(0)

# start_practice()


# ========================================================================================== #
# Завдання 1
# Отримайте перший кадр з файлу data\lesson8\animals.mp4 та виведіть його на екран.
# Проведіть детекцію об’єктів зо допомогою YOLO та виведіть результати.
# Змініть параметри моделі conf та iou і подивіться як це впливає на результат.
# Отримайте рамки для кожного об’єкта, виріжіть їх та виведіть як окремі зображення

def start_task1():
    cap = cv2.VideoCapture('../data/lesson8/animals.mp4')

    success, img = cap.read()
    img = cv2.resize(img, None, fx=0.5, fy=0.5)

    cv2.imshow('original', img)

    results = model.predict(
        img,
        device='mps',
        conf=0.9,
        iou=0.4
    )

    results = results[0]
    boxes = results.boxes

    for i, box in enumerate(boxes):
        xyxy = box.xyxy.cpu().numpy()
        x1, y1, x2, y2 = map(int, xyxy[0])
        roi = img[y1:y2, x1:x2]
        cv2.imshow(f'box {i + 1}', roi)

    r = results.plot()
    cv2.imshow('result', r)

    cv2.waitKey(0)

# start_task1()

# ========================================================================================== #

# Завдання 2
# Напишіть програму по відстеженню об’єкта на відео.
# Відкрийте відео з файлу data\lesson8\animals.mp4 та виведіть на екран результат детекції.
# Попросіть користувача ввести ID об’єкта, який потрібно відслідковувати
# Для вcіх наступних кадрів проведіть детекцію, отримайте
# рамку для об’єкта з потрібним ID та виведіть її на екран.
# Додатково показуйте оригінальне відео.
# Скористайтесь для роботи model.track()

class TrackerObject:
    def __init__(self, path: str):
        self.path = path
        self.cap = cv2.VideoCapture(self.path)
        self.target_id: int | None = None

    def _get_frame(self, img):
        img = cv2.resize(img, None, fx=0.5, fy=0.5)

        results = model.track(
            img,
            device='mps',
            conf=0.9,
            iou=0.4,
            persist=True,
            verbose=False
        )[0]

        return results

    def _select_frame(self, img):
        first_frame = self._get_frame(img)
        boxes = first_frame.boxes

        window_title = 'Оберіть ID обʼєкта для відстеження'
        cv2.imshow( window_title, first_frame.plot())
        cv2.waitKey(1)

        while True:
            try:
                target_id = int(input('Введіть необхідний ID обʼєкта для відстеження: '))

                if target_id <= 0 or target_id > len(boxes):
                    print('ID обʼєкта має бути в межах, що на зображенні')
                    continue

                self.target_id = target_id
                cv2.destroyWindow(window_title)
                break

            except ValueError:
                print('ID обʼєкта має бути цілим числом')
                continue

    def _show_frame(self, img):
        boxes = self._get_frame(img).boxes

        if self.target_id is None:
            return

        id = self.target_id - 1
        xyxy = boxes[id].xyxy.cpu().numpy()
        x1, y1, x2, y2 = map(int, xyxy[0])
        roi = img[y1:y2, x1:x2]
        cv2.imshow('frame', roi)

    def _cleanup(self):
        self.cap.release()
        cv2.destroyAllWindows()

    def track(self):
        success, img = self.cap.read()
        self._select_frame(img)
        cv2.imshow('origin', self._get_frame(img).plot())

        while True:
            success, img = self.cap.read()

            if not success:
                break

            self._show_frame(img)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        self._cleanup()

tracker = TrackerObject('../data/lesson8/animals.mp4')
tracker.track()