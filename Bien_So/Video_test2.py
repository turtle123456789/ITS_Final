import math

import cv2
import numpy as np
import Preprocess


ADAPTIVE_THRESH_BLOCK_SIZE = 19
ADAPTIVE_THRESH_WEIGHT = 9

Min_char_area = 0.015
Max_char_area = 0.06

Min_char = 0.01
Max_char = 0.09

Min_ratio_char = 0.25
Max_ratio_char = 0.7

max_size_plate = 18000
min_size_plate = 5000

RESIZED_IMAGE_WIDTH = 20
RESIZED_IMAGE_HEIGHT = 30

# Load KNN model
npaClassifications = np.loadtxt("classifications.txt", np.float32)
npaFlattenedImages = np.loadtxt("flattened_images.txt", np.float32)
npaClassifications = npaClassifications.reshape(
    (npaClassifications.size, 1))  # Dua ve dang 1D
kNearest = cv2.ml.KNearest_create()  # Khoi tao KNN
kNearest.train(npaFlattenedImages, cv2.ml.ROW_SAMPLE, npaClassifications)

cap = cv2.VideoCapture("7866043685500506488.mp4")
while (cap.isOpened()):
    tongframe = 0
    biensotimthay = 0

    ret, img = cap.read()
    tongframe = tongframe + 1
    # img = cv2.resize(img, None, fx=0.5, fy=0.5)
    imgGrayscaleplate, imgThreshplate = Preprocess.preprocess(img)
    canny_image = cv2.Canny(imgThreshplate, 250, 255)
    kernel = np.ones((3, 3), np.uint8)
    dilated_image = cv2.dilate(canny_image, kernel,iterations=1)  # Dilation

    # Loc bien so
    contours, hierarchy = cv2.findContours(dilated_image, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]  # Lay 10 contour max
    screenCnt = []
    for c in contours:
        peri = cv2.arcLength(c, True)  # Tinh chu vi
        approx = cv2.approxPolyDP(c, 0.06 * peri, True)  # tinh xap xi duong vien
        [x, y, w, h] = cv2.boundingRect(approx.copy())
        ratio = w / h
        if (len(approx) == 4) and (0.8 <= ratio <= 1.5 or 4.5 <= ratio <= 6.5):
            screenCnt.append(approx)
    if screenCnt is None:
        detected = 0
        print("No detected")
    else:
        detected = 1

    if detected == 1:
        n = 1
        for screenCnt in screenCnt:

            #tim goc cua bien so
            (x1, y1) = screenCnt[0, 0]
            (x2, y2) = screenCnt[1, 0]
            (x3, y3) = screenCnt[2, 0]
            (x4, y4) = screenCnt[3, 0]
            array = [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
            sorted_array = array.sort(reverse=True, key=lambda x: x[1])
            (x1, y1) = array[0]
            (x2, y2) = array[1]

            doi = abs(y1 - y2)
            ke = abs(x1 - x2)
            angle = math.atan(doi / ke) * (180.0 / math.pi)

            # Che phan khac voi bien so
            mask = np.zeros(imgGrayscaleplate.shape, np.uint8)
            new_image = cv2.drawContours(mask, [screenCnt], 0, 255, -1, )

            # Now crop
            (x, y) = np.where(mask == 255)
            (topx, topy) = (np.min(x), np.min(y))
            (bottomx, bottomy) = (np.max(x), np.max(y))

            roi = img[topx:bottomx + 1, topy:bottomy + 1]
            imgThresh = imgThreshplate[topx:bottomx + 1, topy:bottomy + 1]

            ptPlateCenter = (bottomx - topx) / 2, (bottomy - topy) / 2

            if x1 < x2:
                rotationMatrix = cv2.getRotationMatrix2D(ptPlateCenter, -angle, 1.0)
            else:
                rotationMatrix = cv2.getRotationMatrix2D(ptPlateCenter, angle, 1.0)

            roi = cv2.warpAffine(roi, rotationMatrix, (bottomy - topy, bottomx - topx))
            imgThresh = cv2.warpAffine(imgThresh, rotationMatrix, (bottomy - topy, bottomx - topx))

            roi = cv2.resize(roi, (0, 0), fx=3, fy=3)
            imgThresh = cv2.resize(imgThresh, (0, 0), fx=3, fy=3)

            #Tien xu ly
            kerel3 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            thre_mor = cv2.morphologyEx(imgThresh, cv2.MORPH_DILATE, kerel3)
            cont, hier = cv2.findContours(thre_mor, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # Phan doan doi tuong
            char_x_ind = {}
            char_x = []
            height, width, _ = roi.shape
            roiarea = height * width

            for ind, cnt in enumerate(cont):
                area = cv2.contourArea(cnt)
                (x, y, w, h) = cv2.boundingRect(cont[ind])
                ratiochar = w / h
                if (Min_char * roiarea < area < Max_char * roiarea) and (0.25 < ratiochar < 0.7):
                    if x in char_x:  # SD du cho trung x van ve dc
                        x = x + 1
                    char_x.append(x)
                    char_x_ind[x] = ind

            # Nhan dang
            if len(char_x) in range(7, 10):
                cv2.drawContours(img, [screenCnt], -1, (0, 255, 0), 3)

                char_x = sorted(char_x)
                strFinalString = ""
                first_line = ""
                second_line = ""

                for i in char_x:
                    (x, y, w, h) = cv2.boundingRect(cont[char_x_ind[i]])
                    cv2.rectangle(roi, (x, y), (x + w, y + h), (0, 255, 0), 2)

                    imgROI = thre_mor[y:y + h, x:x + w]  # crop

                    imgROIResized = cv2.resize(imgROI,
                                            (RESIZED_IMAGE_WIDTH, RESIZED_IMAGE_HEIGHT))  # resize
                    npaROIResized = imgROIResized.reshape(
                        (1, RESIZED_IMAGE_WIDTH * RESIZED_IMAGE_HEIGHT))  # dua anh ve mang 1 chieu
                    # chuyen anh ve dang ma tran
                    npaROIResized = np.float32(npaROIResized)  # Dang float
                    _, npaResults, neigh_resp, dists = kNearest.findNearest(npaROIResized,k=3)  # call KNN function find_nearest; neigh_resp là hàng xóm
                    strCurrentChar = str(chr(int(npaResults[0][0])))  # ASCII
                    cv2.putText(roi, strCurrentChar, (x, y + 50), cv2.FONT_HERSHEY_DUPLEX, 2, (0, 255, 255), 3)

                    if (y < height / 3):   # decide 1 or 2-line license plate
                        first_line = first_line + strCurrentChar
                    else:
                        second_line = second_line + strCurrentChar

                strFinalString = first_line + second_line

                print("\nXe: " + strFinalString + " Đã vượt đèn đỏ\n")

                cv2.putText(img, strFinalString, (topy, topx), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 255, 255), 1)
                n = n + 1
                biensotimthay = biensotimthay + 1

    imgcopy = cv2.resize(img, None, fx=0.5, fy=0.5)
    cv2.imshow('Bien so xe', imgcopy)
    if cv2.waitKey(100) == ord('q'): #nhan phim 'q' de thoat
        break
cap.release()
# cv2.destroyAllWindows()
# img = cv2.imread("data/image/3.jpg")
# # img = cv2.resize(img, dsize=(1920, 1080))
#
# # img = cv2.resize(img, dsize=(1080, 1440))
# # img = cv2.resize(img, dsize=(1920,2560))
# # n = 1
# gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# cv2.imshow("anh_2",gray_image)
#
# # img2 = cv2.imread("1.jpg")
# imgGrayscaleplate2, _ = Preprocess.preprocess(img)
# imgThreshplate2 = cv2.adaptiveThreshold(imgGrayscaleplate2, 250, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,
#                                         ADAPTIVE_THRESH_BLOCK_SIZE, ADAPTIVE_THRESH_WEIGHT)
# cv2.imshow("imgThreshplate2", imgThreshplate2)
# # Xu ly img
# imgGrayscaleplate, imgThreshplate = Preprocess.preprocess(img)
# canny_image = cv2.Canny(imgThreshplate, 250, 255)  # Canny Edge
# kernel = np.ones((3, 3), np.uint8)
# dilated_image = cv2.dilate(canny_image, kernel, iterations=1)  # Dilation
# cv2.imshow("dilated_image", dilated_image)
#
# # Ve duong vien va loc bien so
# contours, hierarchy = cv2.findContours(dilated_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
# contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]  # Lay 10 contour max
# # cv2.drawContours(img, contours, -1, (255, 0, 255), 3) # ve tat ca contour
#
# screenCnt = []
# for c in contours:
#     peri = cv2.arcLength(c, True)  # Tinh chu vi
#     approx = cv2.approxPolyDP(c, 0.06 * peri, True)  # lam xap xi da giac, giu contour 4 canh
#     [x, y, w, h] = cv2.boundingRect(approx.copy())
#     ratio = w / h
#     # cv2.putText(img, str(len(approx.copy())), (x,y),cv2.FONT_HERSHEY_DUPLEX, 2, (0, 255, 0), 3)
#     # cv2.putText(img, str(ratio), (x,y),cv2.FONT_HERSHEY_DUPLEX, 2, (0, 255, 0), 3)
#     if (len(approx) == 4):
#         screenCnt.append(approx)
#
#         cv2.putText(img, str(len(approx.copy())), (x, y), cv2.FONT_HERSHEY_DUPLEX, 2, (0, 255, 0), 2)
#
# if screenCnt is None:
#     detected = 0
#     print("No detected")
# else:
#     detected = 1
#
# if detected == 1:
#
#     for screenCnt in screenCnt:
#         cv2.drawContours(img, [screenCnt], 0, (0, 255, 0), 3)  # Khoanh vùng biển số xe
#
#         # Tim goc bien so
#         (x1, y1) = screenCnt[0, 0]
#         (x2, y2) = screenCnt[1, 0]
#         (x3, y3) = screenCnt[2, 0]
#         (x4, y4) = screenCnt[3, 0]
#         array = [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
#         sorted_array = array.sort(reverse=True, key=lambda x: x[1])
#         (x1, y1) = array[0]
#         (x2, y2) = array[1]
#         doi = abs(y1 - y2)
#         ke = abs(x1 - x2)
#         angle = math.atan(doi / ke) * (180.0 / math.pi)
#
#         # Cat va can chinh theo goc
#         mask = np.zeros(imgGrayscaleplate.shape, np.uint8)
#         new_image = cv2.drawContours(mask, [screenCnt], 0, 255, -1, )
#         # cv2.imshow("new_image",new_image)
#
#         #
#         (x, y) = np.where(mask == 255)
#         (topx, topy) = (np.min(x), np.min(y))
#         (bottomx, bottomy) = (np.max(x), np.max(y))
#
#         roi = img[topx:bottomx, topy:bottomy]
#         imgThresh = imgThreshplate[topx:bottomx, topy:bottomy]
#         ptPlateCenter = (bottomx - topx) / 2, (bottomy - topy) / 2
#
#         if x1 < x2:
#             rotationMatrix = cv2.getRotationMatrix2D(ptPlateCenter, -angle, 1.0)
#         else:
#             rotationMatrix = cv2.getRotationMatrix2D(ptPlateCenter, angle, 1.0)
#
#         roi = cv2.warpAffine(roi, rotationMatrix, (bottomy - topy, bottomx - topx))
#         imgThresh = cv2.warpAffine(imgThresh, rotationMatrix, (bottomy - topy, bottomx - topx))
#         roi = cv2.resize(roi, (0, 0), fx=3, fy=3)
#         imgThresh = cv2.resize(imgThresh, (0, 0), fx=3, fy=3)
#
#         # Tien xu ly
#         kerel3 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
#         thre_mor = cv2.morphologyEx(imgThresh, cv2.MORPH_DILATE, kerel3)
#         cont, hier = cv2.findContours(thre_mor, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#
#         cv2.imshow("Anh loc", thre_mor)
#         cv2.drawContours(roi, cont, -1, (100, 255, 255), 2)
#
#         # Loc cai ki tu
#         char_x_ind = {}
#         char_x = []
#         height, width, _ = roi.shape
#         roiarea = height * width
#
#         for ind, cnt in enumerate(cont):
#             (x, y, w, h) = cv2.boundingRect(cont[ind])
#             ratiochar = w / h
#             char_area = w * h
#             # cv2.putText(roi, str(char_area), (x, y+20),cv2.FONT_HERSHEY_DUPLEX, 2, (255, 255, 0), 2)
#             # cv2.putText(roi, str(ratiochar), (x, y+20),cv2.FONT_HERSHEY_DUPLEX, 2, (255, 255, 0), 2)
#
#             if (Min_char * roiarea < char_area < Max_char * roiarea) and (0.25 < ratiochar < 0.7):
#                 if x in char_x:
#                     x = x + 1
#                 char_x.append(x)
#                 char_x_ind[x] = ind
#
#                 # cv2.putText(roi, str(char_area), (x, y+20),cv2.FONT_HERSHEY_DUPLEX, 2, (255, 255, 0), 2)
#
#         # Nhan dang
#         char_x = sorted(char_x)
#         strFinalString = ""
#         first_line = ""
#         second_line = ""
#
#         for i in char_x:
#             (x, y, w, h) = cv2.boundingRect(cont[char_x_ind[i]])
#             cv2.rectangle(roi, (x, y), (x + w, y + h), (0, 255, 0), 2)
#
#             imgROI = thre_mor[y:y + h, x:x + w]  # Crop
#
#             imgROIResized = cv2.resize(imgROI, (RESIZED_IMAGE_WIDTH, RESIZED_IMAGE_HEIGHT))
#             npaROIResized = imgROIResized.reshape(
#                 (1, RESIZED_IMAGE_WIDTH * RESIZED_IMAGE_HEIGHT))
#
#             npaROIResized = np.float32(npaROIResized)
#             _, npaResults, neigh_resp, dists = kNearest.findNearest(npaROIResized,
#                                                                     k=3)  # call KNN function find_nearest
#             strCurrentChar = str(chr(int(npaResults[0][0])))  # ASCII of characters
#             cv2.putText(roi, strCurrentChar, (x, y + 50), cv2.FONT_HERSHEY_DUPLEX, 2, (255, 255, 0), 3)
#
#             if (y < height / 3):
#                 first_line = first_line + strCurrentChar
#             else:
#                 second_line = second_line + strCurrentChar
#
#         print("\n Bien so xe: " + first_line + " - " + second_line + "\n")
#         roi = cv2.resize(roi, None, fx=0.5, fy=0.5)
#         cv2.imshow("Nhan dang", cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))
#
#         cv2.putText(img, first_line + "-" + second_line, (topy, topx), cv2.FONT_HERSHEY_DUPLEX, 2, (0, 255, 255), 2)
#         # n = n + 1
#
# img = cv2.resize(img, None, fx=0.5, fy=0.5)
# # img = cv2.resize(img, None, fx=0.75, fy=0.75)
# cv2.imshow('Bien so xe', img)
# cv2.waitKey()