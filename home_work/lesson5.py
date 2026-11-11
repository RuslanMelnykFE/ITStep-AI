import cv2

#===============================================================================================================#
# Відкрийте відео з файлу data\lesson7\meter.mp4.
# Проведіть бінарізацію кадрів та збережіть в новий файл.
# Можливо очистіть від шуму або наведіть різкість через bilateralFilter


def start_task():
    cap = cv2.VideoCapture('../data/lesson7/meter.mp4')
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    result = cv2.VideoWriter(
        '../data/lesson7/output_meter.avi',
        fourcc,
        fps,
        (width, height))


    while True:
        success, frame = cap.read()
        if not success:
            break

        cv2.imshow('frame', frame)

        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        frame_binary = cv2.adaptiveThreshold(
            frame_gray,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            21,
            1.8
        )

        frame_bf = cv2.bilateralFilter(
            frame_binary,
            d=9,
            sigmaColor=25,
            sigmaSpace=55
        )

        frame_res = cv2.fastNlMeansDenoising(
            frame_bf,
            None,
            h=20,
            templateWindowSize=15,
            searchWindowSize=54
        )

        cv2.imshow("Відео", frame_res)

        result.write(frame_res)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    result.release()
    cv2.destroyAllWindows()

start_task()

