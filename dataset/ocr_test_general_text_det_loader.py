#coding:utf-8
# dataloader add 3.0 scale
# dataloader add filer text
import numpy as np
from PIL import Image
from torch.utils import data
import util
import cv2
import random
import torchvision.transforms as transforms
import torch
import pyclipper
import Polygon as plg

ocr_root_dir = './testimg/'
ocr_test_data_dir = ocr_root_dir + 'mark/'
ocr_test_gt_dir = ocr_root_dir + 'text_label_curve/'
# ocr_test_data_dir = ocr_root_dir + 'train/text_image/'
# ocr_test_gt_dir = ocr_root_dir + 'train/text_label_curve/'

random.seed(123456)

def get_img(img_path):
    try:
        img = cv2.imread(img_path)
        img = img[:, :, [2, 1, 0]]
    except Exception as e:
        print img_path
        raise
    return img

def get_img1(img):
    try:
        img = img[:, :, [2, 1, 0]]
    except Exception as e:
        print "Img Error"
        raise
    return img

def get_bboxes(img, gt_path):
    h, w = img.shape[0:2]
    lines = util.io.read_lines(gt_path)
    bboxes = []
    tags = []
    for line in lines:
        line = util.str.remove_all(line, '\xef\xbb\xbf')
        gt = util.str.split(line, ',')

        # x1 = np.int(gt[0])
        # y1 = np.int(gt[1])
        x1 = np.int(float(gt[0]))
        y1 = np.int(float(gt[1]))

        #bbox = [np.int(gt[i]) for i in range(4, 32)]
        bbox = [np.int(float(gt[i])) for i in range(4, 32)]
        bbox = np.asarray(bbox) + ([x1 * 1.0, y1 * 1.0] * 14)
        bbox = np.asarray(bbox) / ([w * 1.0, h * 1.0] * 14)
        
        bboxes.append(bbox)
        tags.append(True)
    return np.array(bboxes), tags

def random_horizontal_flip(imgs):
    if random.random() < 0.5:
        for i in range(len(imgs)):
            imgs[i] = np.flip(imgs[i], axis=1).copy()
    return imgs

def random_rotate(imgs):
    max_angle = 10
    angle = random.random() * 2 * max_angle - max_angle
    for i in range(len(imgs)):
        img = imgs[i]
        w, h = img.shape[:2]
        rotation_matrix = cv2.getRotationMatrix2D((h / 2, w / 2), angle, 1)
        img_rotation = cv2.warpAffine(img, rotation_matrix, (h, w))
        imgs[i] = img_rotation
    return imgs

def scale(img, long_size=1280):
    h, w = img.shape[0:2]
    scale = long_size * 1.0 / max(h, w)
    img = cv2.resize(img, dsize=None, fx=scale, fy=scale)
    ### insert by yuyang 解决定位偏移问题
    resize_h, resize_w = img.shape[0:2]
    #print("new_h: {0}, new_w: {1}".format(resize_h, resize_w))
    resize_h = resize_h if resize_h % 32 == 0 else (resize_h // 32 + 1) * 32
    resize_w = resize_w if resize_w % 32 == 0 else (resize_w // 32 + 1) * 32
    img = cv2.resize(img, (int(resize_w), int(resize_h)))
    #print("resize_h: {0}, resize_w: {1}".format(resize_h, resize_w))
    ###
    return img

def random_scale(img, min_size):
    h, w = img.shape[0:2]
    if max(h, w) > 1280:
        scale = 1280.0 / max(h, w)
        img = cv2.resize(img, dsize=None, fx=scale, fy=scale)

    h, w = img.shape[0:2]
    random_scale = np.array([0.5, 1.0, 2.0, 3.0])
    scale = np.random.choice(random_scale)
    if min(h, w) * scale <= min_size:
        scale = (min_size + 10) * 1.0 / min(h, w)
    img = cv2.resize(img, dsize=None, fx=scale, fy=scale)
    return img

def random_crop(imgs, img_size):
    h, w = imgs[0].shape[0:2]
    th, tw = img_size
    if w == tw and h == th:
        return imgs
    
    if random.random() > 3.0 / 8.0 and np.max(imgs[1]) > 0:
        tl = np.min(np.where(imgs[1] > 0), axis = 1) - img_size
        tl[tl < 0] = 0
        br = np.max(np.where(imgs[1] > 0), axis = 1) - img_size
        br[br < 0] = 0
        br[0] = min(br[0], h - th)
        br[1] = min(br[1], w - tw)
        
        i = random.randint(tl[0], br[0])
        j = random.randint(tl[1], br[1])
    else:
        i = random.randint(0, h - th)
        j = random.randint(0, w - tw)
    
    # return i, j, th, tw
    for idx in range(len(imgs)):
        if len(imgs[idx].shape) == 3:
            imgs[idx] = imgs[idx][i:i + th, j:j + tw, :]
        else:
            imgs[idx] = imgs[idx][i:i + th, j:j + tw]
    return imgs

def dist(a, b):
    return np.sqrt(np.sum((a - b) ** 2))

def perimeter(bbox):
    peri = 0.0
    for i in range(bbox.shape[0]):
        peri += dist(bbox[i], bbox[(i + 1) % bbox.shape[0]])
    return peri

def shrink(bboxes, rate, max_shr=20):
    rate = rate * rate
    shrinked_bboxes = []
    for bbox in bboxes:
        area = plg.Polygon(bbox).area()
        peri = perimeter(bbox)

        pco = pyclipper.PyclipperOffset()
        pco.AddPath(bbox, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
        offset = min((int)(area * (1 - rate) / (peri + 0.001) + 0.5), max_shr)
        
        shrinked_bbox = pco.Execute(-offset)
        if len(shrinked_bbox) == 0:
            shrinked_bboxes.append(bbox)
            continue
        
        shrinked_bbox = np.array(shrinked_bbox[0])
        if shrinked_bbox.shape[0] <= 2:
            shrinked_bboxes.append(bbox)
            continue
        
        shrinked_bboxes.append(shrinked_bbox)
    
    return np.array(shrinked_bboxes)

class OCRTestGeneralTextDetLoader(data.Dataset):
    def __init__(self, long_size=1280, is_transform=False, img_size=None, kernel_num=7, min_scale=0.4):
        self.is_transform = is_transform
        
        self.img_size = img_size if (img_size is None or isinstance(img_size, tuple)) else (img_size, img_size)
        self.kernel_num = kernel_num
        self.min_scale = min_scale
        
        data_dirs = [ocr_test_data_dir]
        gt_dirs = [ocr_test_gt_dir]
        
        self.img_paths = []
        self.gt_paths = []
        
        for data_dir, gt_dir in zip(data_dirs, gt_dirs):
            img_names = util.io.ls(data_dir, '.jpg')
            img_names.extend(util.io.ls(data_dir, '.png'))
            # img_names.extend(util.io.ls(data_dir, '.gif'))

            img_paths = []
            gt_paths = []
            for idx, img_name in enumerate(img_names):
                img_path = data_dir + img_name
                img_paths.append(img_path)
                
                #insert by yuyang
                if img_name.find('.jpg') != -1:
                    gt_name = img_name.split('.jpg')[0] + '.txt'
                elif img_name.find('.png') != -1:
                    gt_name = img_name.split('.png')[0] + '.txt'
                elif img_name.find('.bmp') != -1:
                    gt_name = img_name.split('.bmp')[0] + '.txt'
                else:
                    gt_name = img_name + '.txt'

                #gt_name = img_name.split('.')[0] + '.txt'
                gt_path = gt_dir + gt_name
                gt_paths.append(gt_path)

            self.img_paths.extend(img_paths)
            self.gt_paths.extend(gt_paths)
            
        # self.img_paths = self.img_paths[440:]
        # self.gt_paths = self.gt_paths[440:]
        #self.long_size = long_size
        #self.long_size = 2240

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, index):
        img_path = self.img_paths[index]
        gt_path = self.gt_paths[index]

        img = get_img(img_path)
        src_img = get_img(img_path)
        bboxes, tags = get_bboxes(img, gt_path)

        if self.is_transform:
            img = random_scale(img, self.img_size[0])
        
        gt_text = np.zeros(img.shape[0:2], dtype='uint8')
        training_mask = np.ones(img.shape[0:2], dtype='uint8')
        if bboxes.shape[0] > 0:
            bboxes = np.reshape(bboxes * ([img.shape[1], img.shape[0]] * 14), (bboxes.shape[0], bboxes.shape[1] / 2, 2)).astype('int32')
            for i in range(bboxes.shape[0]):
                cv2.drawContours(gt_text, [bboxes[i]], -1, i + 1, -1)
                if not tags[i]:
                    cv2.drawContours(training_mask, [bboxes[i]], -1, 0, -1)

        gt_kernals = []
        for i in range(1, self.kernel_num):
            rate = 1.0 - (1.0 - self.min_scale) / (self.kernel_num - 1) * i
            gt_kernal = np.zeros(img.shape[0:2], dtype='uint8')
            kernal_bboxes = shrink(bboxes, rate)
            for i in range(bboxes.shape[0]):
                cv2.drawContours(gt_kernal, [kernal_bboxes[i]], -1, 1, -1)
            gt_kernals.append(gt_kernal)

        if self.is_transform:
            imgs = [img, gt_text, training_mask]
            imgs.extend(gt_kernals)

            imgs = random_horizontal_flip(imgs)
            imgs = random_rotate(imgs)
            imgs = random_crop(imgs, self.img_size)

            img, gt_text, training_mask, gt_kernals = imgs[0], imgs[1], imgs[2], imgs[3:]

        gt_text[gt_text > 0] = 1
        gt_kernals = np.array(gt_kernals)

        # '''
        if self.is_transform:
            img = Image.fromarray(img)
            img = img.convert('RGB')
            img = transforms.ColorJitter(brightness = 32.0 / 255, saturation = 0.5)(img)
        else:
            img = Image.fromarray(img)
            img = img.convert('RGB')

        img = transforms.ToTensor()(img)
        img = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(img)

        gt_text = torch.from_numpy(gt_text).float()
        gt_kernals = torch.from_numpy(gt_kernals).float()
        training_mask = torch.from_numpy(training_mask).float()

        # scaled_img = scale(src_img, self.long_size)
        # scaled_img = Image.fromarray(scaled_img)
        # scaled_img = scaled_img.convert('RGB')
        # scaled_img = transforms.ToTensor()(scaled_img)
        # scaled_img = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(scaled_img)
        
        #return img[:, :, [2, 1, 0]], scaled_img
        return img[:, :, [2, 1, 0]], img, gt_text, gt_kernals, training_mask
        #return src_img[:, :, [2, 1, 0]], scaled_img, gt_text, gt_kernals, training_mask

class OCRTestGeneralTextDetLoader_1(data.Dataset):
    def __init__(self, long_size=1280):
        
        data_dirs = [ocr_test_data_dir]
        
        self.img_paths = []
        
        for data_dir in data_dirs:
            img_names = util.io.ls(data_dir, '.jpg')
            img_names.extend(util.io.ls(data_dir, '.png'))
            # img_names.extend(util.io.ls(data_dir, '.gif'))

            img_paths = []
            for idx, img_name in enumerate(img_names):
                img_path = data_dir + img_name
                img_paths.append(img_path)

            self.img_paths.extend(img_paths)
            
        # self.img_paths = self.img_paths[440:]
        # self.gt_paths = self.gt_paths[440:]
        self.long_size = long_size

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, index):
        img_path = self.img_paths[index]

        img = get_img(img_path)

        scaled_img = scale(img, self.long_size)
        scaled_img = Image.fromarray(scaled_img)
        scaled_img = scaled_img.convert('RGB')
        scaled_img = transforms.ToTensor()(scaled_img)
        scaled_img = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(scaled_img)
        
        return img[:, :, [2, 1, 0]], scaled_img

class OCRTestGeneralTextDetLoader_2(data.Dataset):
    #def __init__(self, img_path, long_size=1280):
    def __init__(self, img, long_size=1280):
              
        #self.img_paths = [img_path] 
        self.imgs = [img]

        # self.img_paths = self.img_paths[440:]
        # self.gt_paths = self.gt_paths[440:]
        self.long_size = long_size

    def __len__(self):
        #return len(self.img_paths)
        return len(self.imgs)

    def __getitem__(self, index):
        #img_path = self.img_paths[index]
        #img = get_img(img_path)

        img = self.imgs[index]
        img = get_img1(img)

        scaled_img = scale(img, self.long_size)
        scaled_img = Image.fromarray(scaled_img)
        scaled_img = scaled_img.convert('RGB')
        scaled_img = transforms.ToTensor()(scaled_img)
        scaled_img = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(scaled_img)
        
        return img[:, :, [2, 1, 0]], scaled_img
