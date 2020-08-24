import face_alignment
import cv2
import numpy as np

class FaceDetector():
    def __init__(self, scale=1.1):
        self.bbox = []
        self.landmarks = []
        self.scale = scale
        self.detector = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D)

    def detect_faces(self, img):
        '''
        Detect faces in image
        :param img: cv::mat HxWx3 RGB
        :return: yield 4 <x,y,w,h>
        '''
        self.bbox.clear()
        self.landmarks.clear()
        img_h, img_w = img.shape[:2]
        # detect faces
        landmarks = self.detector.get_landmarks(img)
        bbs = []
        if isinstance(landmarks, type(None)):
            return
        for _landmark in landmarks:
            (left, top), (right, bottom) = _landmark.min(axis=0), _landmark.max(axis=0)
            bbs.append([left, top, right, bottom])
            self.landmarks.append(_landmark)

        for region in bbs:
            left, top, right, bottom = region
            center = (top + (bottom - top) / 2, left + (right - left) / 2)
            size = self.scale * (right - left + bottom - top) / 2
            top, bottom = int(center[0] - size / 2), int(center[0] + size / 2)
            left, right = int(center[1] - size / 2), int(center[1] + size / 2)
            # boundary detection
            top = 0 if top < 0 else top
            bottom = img_h if bottom > img_h else bottom
            left = 0 if left < 0 else left
            right = img_w if right > img_w else right
            self.bbox.append(np.array([left, top, right, bottom]))


def crop_face_with_bb(img, bb):
    '''
    Crop face in image given bb
    :param img: cv::mat HxWx3
    :param bb: 4 (<x,y,w,h>)
    :return: HxWx3
    '''
    x, y, w, h = bb
    return img[y:y+h, x:x+w, :]

def place_face(img, face, bb):
    x, y, w, h = bb
    face = resize_face(face, size=(w, h))
    img[y:y+h, x:x+w] = face
    return img

def resize_face(face_img, size=(128, 128)):
    '''
    Resize face to a given size
    :param face_img: cv::mat HxWx3
    :param size: new H and W (size x size). 128 by default.
    :return: cv::mat size x size x 3
    '''
    return cv2.resize(face_img, size)


def generate_crop_box(image_info=None, scale=1.1):
    '''
    giving provided image_info and rescale the box to new size
    Args:
        image_info: the bounding box or the landmarks

    Return:
        a box with 4 values: [left, top, right, bottom] or a
        list contains several box, each has 4 landmarks
    '''
    box = None
    if image_info is not None:
        if np.max(image_info.shape) > 4:  # key points to get bounding box
            kpt = image_info
            if kpt.shape[0] < 3:
                kpt = kpt.T   # nof_marks x 2
            if kpt.shape[0] <= 5:  # 5 x 2
                scale = scale*scale
            left = np.min(kpt[:, 0])
            right = np.max(kpt[:, 0])
            top = np.min(kpt[:, 1])
            bottom = np.max(kpt[:, 1])
        else:  # bounding box
            bbox = image_info
            left = bbox[0]
            right = bbox[2]
            top = bbox[1]
            bottom = bbox[3]

        old_size = (right - left + bottom - top) / 2
        center = np.array([right - (right - left) / 2.0, bottom - (bottom - top) / 2.0])
        size = int(old_size * scale)
        box = [center[0] - size / 2, center[1] - size / 2,
               center[0] + size / 2, center[1] + size / 2]
    return box