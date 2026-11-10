import cv2
from utils import trackbar_decorator


# Завдання 1
# Відкрийте зображення data/lesson3/notes.png. Проведіть наступні дії:
#  проведіть бінарізацію(звичайну та адаптивну)
#  застосуйте розмиття(гаусове) візьміть ядра 3, 5, 11 та sigmaX 0, 2, 10
# повторіть бінарізацію, але перед тим застосуйте bilateral filter

def start_task1():
    img = cv2.imread('../data/lesson3/notes.png')
    cv2.imshow('original', img)

    #  проведіть бінарізацію (звичайну та адаптивну)
    grey_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    binary_img = cv2.threshold(
        grey_img,
        thresh=172,
        maxval=255,
        type=cv2.THRESH_BINARY
    )[1]

    binary_img_a = cv2.adaptiveThreshold(
        grey_img,
        maxValue=255,
        adaptiveMethod=cv2.ADAPTIVE_THRESH_MEAN_C,
        thresholdType=cv2.THRESH_BINARY,
        blockSize=21,
        C=5
    )

    cv2.imshow('binary', binary_img)
    cv2.imshow('binary adaptive', binary_img_a)

    #  застосуйте розмиття(гаусове) візьміть ядра 3, 5, 11 та sigmaX 0, 2, 10
    gaussian_img_3_0 = cv2.GaussianBlur(img, (3, 3), 0)
    gaussian_img_3_2 = cv2.GaussianBlur(img, (3, 3), 2)
    gaussian_img_3_10 = cv2.GaussianBlur(img, (3, 3), 10)

    gaussian_img_5_0 = cv2.GaussianBlur(img, (5, 5), 0)
    gaussian_img_5_2 = cv2.GaussianBlur(img, (5, 5), 2)
    gaussian_img_5_10 = cv2.GaussianBlur(img, (5, 5), 10)

    gaussian_img_11_0 = cv2.GaussianBlur(img, (11, 11), 0)
    gaussian_img_11_2 = cv2.GaussianBlur(img, (11, 11), 2)
    gaussian_img_11_10 = cv2.GaussianBlur(img, (11, 11), 10)

    cv2.imshow('gaussian_img_3_0', gaussian_img_3_0)
    cv2.imshow('gaussian_img_3_2', gaussian_img_3_2)
    cv2.imshow('gaussian_img_3_10', gaussian_img_3_10)

    cv2.imshow('gaussian_img_5_0', gaussian_img_5_0)
    cv2.imshow('gaussian_img_5_2', gaussian_img_5_2)
    cv2.imshow('gaussian_img_5_10', gaussian_img_5_10)

    cv2.imshow('gaussian_img_11_0', gaussian_img_11_0)
    cv2.imshow('gaussian_img_11_2', gaussian_img_11_2)
    cv2.imshow('gaussian_img_11_10', gaussian_img_11_10)

    # повторіть бінарізацію, але перед тим застосуйте bilateral filter

    img_bf = cv2.bilateralFilter(
        grey_img,
        d=9,
        sigmaColor=5,
        sigmaSpace=5
    )

    binary_img_b = cv2.adaptiveThreshold(
        img_bf,
        maxValue=255,
        adaptiveMethod=cv2.ADAPTIVE_THRESH_MEAN_C,
        thresholdType=cv2.THRESH_BINARY,
        blockSize=21,
        C=7
    )

    cv2.imshow('binary adaptive 2', binary_img_b)

    cv2.waitKey(0)

# start_task1()


# Завдання 2
# Відкрийте зображення data/lesson3/sudoku.jpg. Проведіть для нього бінарізацію, а саме
#  CLAHE
#  гаусове розмиття
#  адаптивна бінарізація
#  NLMean
# Самостійно підберіть параметри, збережіть результат.
# Порівняйте результати для гаусової та середньої адаптивної бінарізації

def start_task2():
    img = cv2.imread('../data/lesson3/sudoku.jpg')
    cv2.imshow('original', img)

    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(20, 20))
    l_eq = clahe.apply(l)
    result = cv2.cvtColor(
        cv2.merge([l_eq, a, b]),
        cv2.COLOR_LAB2BGR
    )

    gaussian_img = cv2.GaussianBlur(result, (3, 3), 2)
    grey_img = cv2.cvtColor(gaussian_img, cv2.COLOR_BGR2GRAY)
    adaptive_img = cv2.adaptiveThreshold(
        grey_img,
        maxValue=255,
        adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        thresholdType=cv2.THRESH_BINARY,
        blockSize=21,
        C=15
    )

    result_gray = cv2.fastNlMeansDenoising(
        adaptive_img,
        None,
        h=20,
        templateWindowSize=15,
        searchWindowSize=31
    )


    cv2.imshow('final', result_gray)
    cv2.waitKey(0)


# start_task2()


# Завдання 3
# Використовуючи utils.trackbar_decorator Побудуйте class
# ThresholdingParameterSelector для підбору параметрів для адаптивної бінарізації.
# Методи:
# o run(img, **kwargs) – головний метод який
# запускає застосовує усі перетворення до
# зображення і повертає маску.
# Решту парметрів підберіть самостійно
# Можливі методи:
# o _apply_blur(img, ksize, sigmaX)
# o _apply_bilateral(img, d, sigmaS, sigmaC)
# o _apply_threshold(img, ksize, C)
# o _apply_denoising(img, h, search_size, tamplate_size)

class ThresholdingParameterSelector:
    def _apply_blur(self, img, k_size, sigma_x):
        ksize = k_size

        if ksize % 2 == 0:
            ksize = ksize + 1

        if k_size < 3:
            ksize = 3

        return cv2.GaussianBlur(
            img,
            (ksize, ksize),
            sigmaX=sigma_x
        )

    def _apply_bilateral(self, img, d, sigma_c, sigma_s):
        img_grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        return cv2.bilateralFilter(
            img_grey,
            d=d,
            sigmaColor=sigma_c,
            sigmaSpace=sigma_s
        )

    def _apply_threshold(self, img, k_size, c):
        ksize = k_size

        if ksize % 2 == 0:
            ksize = ksize + 1

        if k_size < 3:
            ksize = 3

        return cv2.adaptiveThreshold(
            img,
            maxValue=255,
            adaptiveMethod=cv2.ADAPTIVE_THRESH_MEAN_C,
            thresholdType=cv2.THRESH_BINARY,
            blockSize=ksize,
            C=c
        )

    def _apply_denoising(self, img, h, search_size, template_size):
        return cv2.fastNlMeansDenoising(
            img,
            None,
            h=h,
            templateWindowSize=template_size,
            searchWindowSize=search_size
        )

    @trackbar_decorator(
        ksize=(0, 21),
        sigmaX=(0, 21),
        sigmaS=(0, 255),
        sigmaC=(0, 255),
        d=(1, 21),
        C=(1, 21),
        h=(1, 21)
    )

    def run(self, img, **kwargs):
        img_blur = self._apply_blur(
            img,
            kwargs['ksize'],
            kwargs['sigmaX']
        )

        img_bi = self._apply_bilateral(
            img_blur, kwargs['d'], kwargs['sigmaC'], kwargs['sigmaS']
        )

        img_thresh = self._apply_threshold(img_bi, kwargs['ksize'], kwargs['C'])

        mask = self._apply_denoising(img_thresh, kwargs['h'], kwargs['ksize'], kwargs['d'])

        return mask


def start_task3():
    img = cv2.imread('../data/lesson3/sudoku.jpg')

    ThresholdingParameterSelector().run(img)


start_task3()