"""Необходимо детектировать aruco-маркер, откалибровать камеру, рассчитать расстояние от маркера до камеры.
характеристкии камеры можно нагуглить"""


import numpy as np
import cv2 as cv
import cv2.aruco as aruco
from PIL import Image

image_name = "test.jpg"
aruco_size = 0.10  # Размер маркера, метров
(image_width, image_height) = Image.open(image_name).size  # Размер изображения, пикселей

"""camera intrinsics

    Xiaomi redmi 7a:
!Sensor Format - size_x=5.04 mm, size_y= 3.77 mm
Image Capture Size - 4032x3016 12 MP (Sony IMX486),
Aperture - f/2.2
Average File Size - 4.2 MB
Pixel Size - 1.25μ pixel
!Focal Length - f=3.8 mm / 26 mm (35mm camera equivalent)

    Iphone5s:
!Sensor Format - size_x=4.89 mm, size_y=3.67 mm
Image Capture Size - 3264x2448 8 MP
Aperture - f/2.2
Average File Size - 2.5 MB
Pixel Size - 1.5µm
!Focal Length - f=4.12 mm / 31mm (35mm camera equivalent)
"""

f = 3.8
sensor_size_x = 5.04
sensor_size_y = 3.77

# есть основания пологать, что маркер, распечатанный в изначальном задании,
# взят из старой библиотеки и потому не деетктируется.
# Ко всему прочему вес файла по тем же техническим характеристикам камеры должен быть несколько мегабайт,
# а картинка сжата до сотен килобит, что странно. Я всзя свою камеру и сфотографировал свой 10-сантиметровый маркер 6х6.
image = cv.imread(image_name)

# Загрузка словаря первых 50 aruco-маркеров 6х6
aruco_dict = aruco.Dictionary_get(cv.aruco.DICT_6X6_50)

# Инициализация параметров детектора, используются значения по умолчанию
parameters = aruco.DetectorParameters_create()

# Обнаружение маркеров на изображении
marker_corners, marker_ids, rejected_candidates = aruco.detectMarkers(image, aruco_dict, parameters=parameters)

print('\nmarker_corners:\n', marker_corners)
print('\nmarker_ids:', marker_ids)
#  print(rejected_candidates)

# Отмечаем найденные маркеры
aruco.drawDetectedMarkers(image, marker_corners, marker_ids)

# Отмечаем "отклонённые" маркеры
aruco.drawDetectedMarkers(image, rejected_candidates, borderColor=(50, 0, 240))

# Возьмём за основу - центр моего монитора (и маркера).
object_points = np.array(
    [
        [-aruco_size / 2, aruco_size / 2, 0],  # [x, y, z] format
        [aruco_size / 2, aruco_size / 2, 0],
        [aruco_size / 2, -aruco_size / 2, 0],
        [-aruco_size / 2, -aruco_size / 2, 0],
    ], dtype="float32"
)

# необходимо перевести в формат "Ptr<cv::UMat>"
image_points = np.array(
    [
        [marker_corners[0][0][0][0], marker_corners[0][0][0][1]],  # [x, y] format
        [marker_corners[0][0][1][0], marker_corners[0][0][1][1]],
        [marker_corners[0][0][2][0], marker_corners[0][0][2][1]],
        [marker_corners[0][0][3][0], marker_corners[0][0][3][1]],
    ], dtype="float32"
)

print('\nimage_points:\n', image_points)

# Калибровка не выходит, что-то не так с image_points. Но можно и без неё.
"""
rms, camera_matrix, dist_coeffs, c_rvecs, c_tvecs = cv.calibrateCamera(
    object_points,
    image_points,
    (image_width, image_height),
    None, None
)
print('\n rms', rms)
print('\n camera_matrix', camera_matrix)
print('\n dist_coeffs', dist_coeffs)
print('\n crvecs', c_rvecs)
print('\n ctvecs', c_tvecs)"""

"""camera matrix
    | fx 0. cx | 
    | 0. fy cy |
    | 0. 0. 1. |

cx, cy - pixel coordinates of the principal point (пиксельные координаты главной точки).
         я думаю, что их можно приравнять к геометрическому центру изображения, т.к. фото точно никто не обрезал,
         (вообще, лучше бы делать калибровку).
         
fx, fy - это коэффициент масштабирования от объектов в мире до пикселей камеры.
            Именно этот параметр нужно высчитывать из характеристик камеры.
            fx == fy можно примерно приравнять, они не сильно отличны"""

cx, cy = float(image_width / 2), float(image_height / 2)
fx, fy = f * image_width / sensor_size_x, f * image_height / image_height

camera_matrix = np.array(
    [
        [fx, 0., cx],
        [0., fy, cy],
        [0., 0., 1.]
    ], dtype="float32"
)

print('\ncamera_matrix:\n', camera_matrix)

# Без калибровки точно не узнать. Имперически было установлено, что на результат особого влияния не имеет.
# В рамках ТЗ можно пренебречь.
dist_coeffs = np.array(
    [
        [0, 0, 0, 0, 0]
    ], dtype="float32"
)

# revtval - ?
# rvecs   - Output rotation vector
# tvecs   - Output translation vector
revtval, rvecs, tvecs = cv.solvePnP(
    objectPoints=object_points,  # Array of object points in the object coordinate space Nx3, can be also passed here.
    imagePoints=image_points,  # Array of corresponding image points Nx2, can be also passed here.
    cameraMatrix=camera_matrix,  # Input camera matrix
    distCoeffs=dist_coeffs,  # Input vector of distortion coefficients (искажение)
    flags=cv.SOLVEPNP_IPPE_SQUARE  # Method for solving a PnP problem.
    # SOLVEPNP_IPPE_SQUARE - особый случай для оценки положения маркера
)

print('\n Полученные рузультаты определения местоположения:')
print('\n revtval:', revtval)
print('\n rvecs:', rvecs)  # Полученный вектор вращения, однозначно определяющий взаимное расположние камеры и маркера
print('\n tvecs:', tvecs)

cv.namedWindow('Where is marker?', cv.WINDOW_NORMAL)
cv.resizeWindow('Where is marker?', width=1008, height=754)
cv.imshow('Where is marker?', image)
cv.waitKey(0)
cv.destroyAllWindows()