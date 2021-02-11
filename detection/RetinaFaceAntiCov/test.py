import cv2
import sys
import numpy as np
import datetime
import os
import glob
from retinaface_cov import RetinaFaceCoV
import glob

thresh = 0.8
mask_thresh = 0.2
scales = [640, 1080]

count = 1

gpuid = 0
#detector = RetinaFaceCoV('./model/mnet_cov1', 0, gpuid, 'net3')
detector = RetinaFaceCoV('./model/mnet_cov2', 0, gpuid, 'net3l')

# os.mkdir("res3")

images = glob.glob("second/images/*.jpg")

# specials = [81]


for ind, image in enumerate(images):

    img = cv2.imread(image)
    # print(img.shape)
    im_shape = img.shape
    target_size = scales[0]
    max_size = scales[1]
    im_size_min = np.min(im_shape[0:2])
    im_size_max = np.max(im_shape[0:2])
    #im_scale = 1.0
    #if im_size_min>target_size or im_size_max>max_size:
    im_scale = float(target_size) / float(im_size_min)
    # prevent bigger axis from being more than max_size:
    if np.round(im_scale * im_size_max) > max_size:
        im_scale = float(max_size) / float(im_size_max)

    # print('im_scale', im_scale)

    # scales = [im_scale]
    flip = True

    for c in range(count):
        faces, landmarks = detector.detect(img,
                                           thresh,
                                           scales=[im_scale],
                                           do_flip=flip)

    if faces is not None and len(faces) != 0:
        # print("name", image)
        # print('find', faces.shape[0], 'faces',faces)
        winner = 0
        largest = 0
        w = np.argmax(faces[:, 4])
        for i,face in enumerate(faces):
            if (face[2] - face[0])+(face[3]-face[1]) > largest:
                largest = face[2] - face[0]
                winner = i
        face = faces[winner]
        # for i in range(faces.shape[0]):
        # print('score', faces[i][4])

        # face = faces[i]
        # print(len(face), face, "faceeee")
        box = face[0:4].astype(np.int)
        mask = face[5]
        # print(i, box, mask)
        #color = (255,0,0)
        if mask >= mask_thresh:
            color = (0, 0, 255)
        else:
            color = (0, 255, 0)
        cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), color, 2)
        # cv2.putText(img, str(face[4]*100), (box[0], box[1]),
        #             cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2)
        landmark5 = landmarks[w].astype(np.int)
        #print(landmark.shape)
        for l in range(landmark5.shape[0]):
            color = (255, 0, 0)
            cv2.circle(
                img, (landmark5[l][0], landmark5[l][1]), 1, color, 2)

        # filename = 'res3/' + str(ind) + '.jpg'
        # if ind in specials:
        # print('writing', filename)
        # cv2.imwrite(filename, img)
    else:
        print("faces is none", image)
