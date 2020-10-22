"""Нужно взять исходное видео, написать код,
чтобы оранжевые тапочки на видео стали цвета стены либо белыми,
обратно прислать код и финальное видео"""

import numpy as np
import cv2 as cv

cap = cv.VideoCapture("video.mp4")
while True:
    ret, frame = cap.read()
    if ret:

        # hsv лучще для распознования цветов, конвертируем BGR в HSV
        hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)

        # в hsv красный (коричневый, оранжевыый) цвет располагается в 2 регионах.
        # Создаём две маски для дающих контрасное ч/б изображение,
        # где красный|оранжевый|коричневый - станут белыми, а остальное - чёрным
        mask_1 = cv.inRange(hsv, (0, 100, 85), (20, 255, 255))
        mask_2 = cv.inRange(hsv, (20, 24, 124), (0, 65, 166))
        mask = cv.bitwise_or(mask_1, mask_2)

        # берём белые пиксели и красим их в (255, 255, 255)
        for i in zip(*np.where(mask == 255)):
            frame[i[0], i[1], 0] = 255
            frame[i[0], i[1], 1] = 255
            frame[i[0], i[1], 2] = 255

        # воспроизводим новое видео
        cv.imshow("res", frame)
        if cv.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

cv.destroyAllWindows()
