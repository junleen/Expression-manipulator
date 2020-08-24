import cv2
import matplotlib.pyplot as plt
import numpy as np


def read_cv2_img(path):
    '''
    Read color images
    Args:
        path: Path to image
    return:
        Only returns color images
    '''
    img = cv2.imread(path, -1)
    
    if img is not None:
        if len(img.shape) != 3:
            return None
        
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    return img

def resize(img, dst_size):
    if isinstance(dst_size, int):
        img = cv2.resize(img, (dst_size, dst_size), cv2.INTER_CUBIC)
    elif isinstance(dst_size, tuple) or isinstance(dst_size, list):
        img = cv2.resize(img, (dst_size[0], dst_size[1]), cv2.INTER_CUBIC)

    return img

def show_cv2_img(img, title='img'):
    '''
    Display cv2 image
    :param img: cv::mat
    :param title: title
    :return: None
    '''
    plt.imshow(img)
    plt.title(title)
    plt.axis('off')
    plt.show()
    
    
def show_images_row(imgs, titles, rows=1):
    '''
    Display grid of cv2 images image
    Args:
        imgs: list [cv::mat]
        titles: titles
    return: None
    '''
    assert ((titles is None) or (len(imgs) == len(titles)))
    num_images = len(imgs)
    
    if titles is None:
        titles = ['Image (%d)' % i for i in range(1, num_images + 1)]
    
    fig = plt.figure()
    for n, (image, title) in enumerate(zip(imgs, titles)):
        ax = fig.add_subplot(rows, np.ceil(num_images / float(rows)), n + 1)
        if image.ndim == 2:
            plt.gray()
        plt.imshow(image)
        ax.set_title(title)
        plt.axis('off')
    plt.show()