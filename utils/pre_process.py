import cv2 as cv
import numpy as np
import math
from PIL import Image
import os

CURRENTPATH = os.path.dirname(os.path.dirname(__file__))
PATH = CURRENTPATH + '/data/originData/'
XGROUDPATH = CURRENTPATH + '/data/firstGround/'
TEMPPATH = XGROUDPATH + 'temp/'  # c存放中间生成的图片 结束后会删除
TESTSMILEPATH = XGROUDPATH + 'test/1/'  # 存放第x轮的测试集笑脸图片
TESTNOTSMILEPATH = XGROUDPATH + 'test/0/'  # 存放第x轮的测试集非笑脸图片
TRAINSMILEPATH = XGROUDPATH + 'train/1/'  # 存放第x轮的训练集笑脸图片
TRAINNOTSMILEPATH = XGROUDPATH + 'train/0/'  # 存放第x轮的训练集非笑脸图片


class SampleImg:
    def __init__(self, path, positions, smile):
        self.path = path
        self.positions = [[positions[3], positions[8]],  # left mouth
                          [positions[4], positions[9]]]  # right mouth
        self.smile = smile  # 1 for smiling, 2 for not smiling


# 欧氏距离(Euclidean Distance)
# np.sqrt(np.sum(np.square(vector1-vector2)))
def getEuclideanDistance(p1, p2):
    vector1 = np.array([p1[0], p1[1]])
    vector2 = np.array([p2[0], p2[1]])
    return np.linalg.norm(vector1 - vector2)


# 根据参数，求仿射变换矩阵和变换后的图像。
def ScaleRotateTranslate(image, angle, center=None, new_center=None, scale=None, resample=Image.BICUBIC):
    if (scale is None) and (center is None):
        return image.rotate(angle=angle, resample=resample)
    nx, ny = x, y = center
    sx = sy = 1.0
    if new_center:
        (nx, ny) = new_center
    if scale:
        (sx, sy) = (scale, scale)
    cosine = math.cos(angle)
    sine = math.sin(angle)
    a = cosine / sx
    b = sine / sx
    c = x - nx * a - ny * b
    d = -sine / sy
    e = cosine / sy
    f = y - nx * d - ny * e
    return image.transform(image.size, Image.AFFINE, (a, b, c, d, e, f), resample=resample)


# 根据所给的人脸图像，嘴巴坐标位置，偏移比例，输出的大小，来进行裁剪。
def CropMouth(image, left_mouth, right_mouth, offset_pct=(0.2, 0.2), dest_sz=(32, 32)):
    # get the direction  计算嘴巴的方向。
    eye_direction = (right_mouth[0] - left_mouth[0], right_mouth[1] - left_mouth[1])
    # calc rotation angle in radians  计算旋转的方向弧度。 反正切值
    rotation = -math.atan2(float(eye_direction[1]), float(eye_direction[0]))
    # rotate original around the left eye  # 原图像绕着左眼的坐标旋转。
    rotated_image = ScaleRotateTranslate(image, center=left_mouth, angle=rotation)
    # calculate offsets in original image 计算在原始图像上的偏移。
    offset_h = math.floor(float(offset_pct[0]) * dest_sz[0])
    offset_v = math.floor(float(offset_pct[1]) * dest_sz[1])
    # distance between them  # 计算两嘴之间的距离。
    dist = getEuclideanDistance(left_mouth, right_mouth)
    # calculate the reference mouth-width    计算最后输出的图像左右嘴角之间的距离。
    reference = dest_sz[0] - 2.0 * offset_h
    # scale factor   # 计算尺度因子。
    scale = float(dist) / float(reference)
    # crop the rotated image  # 剪切
    crop_xy = (left_mouth[0] - scale * offset_h, left_mouth[1] - scale * offset_v)  # 起点
    crop_size = (dest_sz[0] * scale, dest_sz[1] * scale)  # 大小
    rotated_image = rotated_image.crop(
        (int(crop_xy[0]), int(crop_xy[1]), int(crop_xy[0] + crop_size[0]), int(crop_xy[1] + crop_size[1])))
    # resize it 重置大小
    rotated_image = rotated_image.resize(dest_sz, Image.ANTIALIAS)
    return rotated_image


# 载入数据的filename
def load_data(filename):
    data_mat = []
    label_mat = []
    fr = open(PATH + filename)
    for line in fr.readlines():  # 逐行读取
        line_arr = line.strip().split(' ')  # 滤除行首行尾空格，以' '作为分隔符，对这行进行分解
        num = np.shape(line_arr)[0]
        if num == 1:
            break
        data_mat.append(SampleImg(line_arr[0], list(map(float, line_arr[1:num - 4])), int(line_arr[num - 3])))
        label_mat.append(int(line_arr[num - 4]))  # 标签
    return data_mat, label_mat


# 加权平均法
def rgb2gray(file):
    img = cv.imread(PATH + file)
    height = img.shape[0]
    width = img.shape[1]
    for i in range(height):
        for j in range(width):
            img[i, j] = 0.1140 * img[i, j][0] + 0.5870 * img[i, j][1] + 0.2989 * img[i, j][2]
    return img


# 图像自适应灰度直方图均衡化处理
def my_clahe(img):
    mri_img = np.array(img)  # 转化为numpy格式
    # normalization
    mri_max = np.amax(mri_img)
    mri_min = np.amin(mri_img)
    mri_img = ((mri_img - mri_min) / (mri_max - mri_min)) * 255
    mri_img = mri_img.astype('uint8')
    clahe = cv.createCLAHE(clipLimit=2.0,
                           tileGridSize=(2, 2))  # 生成自适应均衡化图像 clipLimit参数表示对比度的大小 tileGridSize参数表示每次处理块的大小
    clahe_arr = clahe.apply(mri_img)
    clahed_img = Image.fromarray(clahe_arr)  # 转化为img格式
    return clahed_img


# 图像预处理 image_obj 自定义的图像对象 test=1代表当前存入测试文件夹
def image_pre_process(image_obj, test):
    grayImg = rgb2gray(image_obj.path)
    image_obj.temp_path = TEMPPATH + os.path.split(image_obj.path)[1]
    cv.imwrite(image_obj.temp_path, grayImg, [int(cv.IMWRITE_JPEG_QUALITY), 100])
    gray_img = Image.open(image_obj.temp_path)
    croppedImg = CropMouth(gray_img, image_obj.positions[0], image_obj.positions[1], (0.15, 0.40), (32, 20))
    # croppedImg.save(image_obj.cropped_path)
    clahedImg = my_clahe(croppedImg.convert("L"))  # 八位像素 黑白
    if test == 1:
        if image_obj.smile == 1:
            clahedImg.save(TESTSMILEPATH + os.path.split(image_obj.path)[1])
        else:
            clahedImg.save(TESTNOTSMILEPATH + os.path.split(image_obj.path)[1])
    if test == 0:
        if image_obj.smile == 1:
            clahedImg.save(TRAINSMILEPATH + os.path.split(image_obj.path)[1])
        else:
            clahedImg.save(TRAINNOTSMILEPATH + os.path.split(image_obj.path)[1])
    os.remove(image_obj.temp_path)
    # clahedImg.save(image_obj.clahed_path)
    # cropped_clahed_img.save(image_obj.cropped_clahed_img)
    return image_obj


# dataMat, labelMat = load_data('testing.txt')
# for i in range(0, np.shape(dataMat)[0]):
#     image_pre_process(dataMat[i], 1)
#
# dataMat, labelMat = load_data('training.txt')
# for i in range(0, np.shape(dataMat)[0]):
#     image_pre_process(dataMat[i], 0)
