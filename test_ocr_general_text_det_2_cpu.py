
#coding:utf-8
import os
import cv2
import sys
import time
import collections
import torch
import torchvision
import argparse
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable
from torch.utils import data

from dataset import OCRTestGeneralTextDetLoader_1
from dataset import OCRTestGeneralTextDetLoader_2

import models
import util
# c++ version pse based on opencv 3+
from pse import pse
# python pse
# from pypse import pse as pypse

###insert by yuyang###
import json
from PIL import Image
import torchvision.transforms as transforms
######################

def extend_3c(img):
    img = img.reshape(img.shape[0], img.shape[1], 1)
    img = np.concatenate((img, img, img), axis=2)
    return img

def debug(idx, img_paths, imgs, output_root):
    if not os.path.exists(output_root):
        os.makedirs(output_root)
    
    col = []
    for i in range(len(imgs)):
        row = []
        for j in range(len(imgs[i])):
            # img = cv2.copyMakeBorder(imgs[i][j], 3, 3, 3, 3, cv2.BORDER_CONSTANT, value=[255, 0, 0])
            row.append(imgs[i][j])
        res = np.concatenate(row, axis=1)
        col.append(res)
    res = np.concatenate(col, axis=0)
    img_name = img_paths[idx].split('/')[-1]
    #print idx, '/', len(img_paths), img_name
    print (idx, '/', len(img_paths), img_name)
    cv2.imwrite(output_root + img_name, res)

def write_result_as_txt(image_name, bboxes, path):
    if not os.path.exists(path):
        os.makedirs(path)

    filename = util.io.join_path(path, '%s.txt'%(image_name))
    lines = []
    for b_idx, bbox in enumerate(bboxes):
        values = [int(v) for v in bbox]
        # line = "%d, %d, %d, %d, %d, %d, %d, %d\n"%tuple(values)
        line = "%d"%values[0]
        for v_id in range(1, len(values)):
            line += ", %d"%values[v_id]
        line += '\n'
        lines.append(line)
    util.io.write_lines(filename, lines)

###inert by yuyang###
def write_result_as_json(image_name, coordinates, path):
    if not os.path.exists(path):
        os.makedirs(path)

    filename = util.io.join_path(path, '%s.jpg.json'%(image_name))
    with open(filename, 'w') as f:
        json.dump(coordinates, f)
#####################

def polygon_from_points(points):
    """
    Returns a Polygon object to use with the Polygon2 class from a list of 8 points: x1,y1,x2,y2,x3,y3,x4,y4
    """
    resBoxes=np.empty([1, 8],dtype='int32')
    resBoxes[0, 0] = int(points[0])
    resBoxes[0, 4] = int(points[1])
    resBoxes[0, 1] = int(points[2])
    resBoxes[0, 5] = int(points[3])
    resBoxes[0, 2] = int(points[4])
    resBoxes[0, 6] = int(points[5])
    resBoxes[0, 3] = int(points[6])
    resBoxes[0, 7] = int(points[7])
    pointMat = resBoxes[0].reshape([2, 4]).T
    return plg.Polygon(pointMat)

def load_pse_model(args):
    # Setup Model
    if args.arch == "resnet50":
        model = models.resnet50(pretrained=True, num_classes=7, scale=args.scale)
    elif args.arch == "resnet101":
        model = models.resnet101(pretrained=True, num_classes=7, scale=args.scale)
    elif args.arch == "resnet152":
        model = models.resnet152(pretrained=True, num_classes=7, scale=args.scale)
    
    for param in model.parameters():
        param.requires_grad = False

    #model = model.cuda()
    
    if args.resume is not None:                                         
        if os.path.isfile(args.resume):
            print("Loading model and optimizer from checkpoint '{}'".format(args.resume))
            #@checkpoint = torch.load(args.resume)
            checkpoint = torch.load(args.resume, map_location='cpu')
            
            # model.load_state_dict(checkpoint['state_dict'])
            d = collections.OrderedDict()
            for key, value in checkpoint['state_dict'].items():
                tmp = key[7:]
                d[tmp] = value
            model.load_state_dict(d)

            print("Loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
            sys.stdout.flush()
        else:
            print("No checkpoint found at '{}'".format(args.resume))
            sys.stdout.flush()

    model.eval()

    return model

def scale(img, long_size=1280):
    h, w = img.shape[0:2]
    scale = long_size * 1.0 / max(h, w)
    img = cv2.resize(img, dsize=None, fx=scale, fy=scale)
    return img
    
#对输入的图片进行预处理
def process_img(img, long_size = 1280):
    scaled_img = scale(img, long_size)
    scaled_img = Image.fromarray(scaled_img)
    scaled_img = scaled_img.convert('RGB')
    scaled_img = transforms.ToTensor()(scaled_img)
    scaled_img = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(scaled_img)
    
    return img[:, :, [2, 1, 0]], scaled_img   

def general_text_detect(input_img, model, long_size = 1280):
    org_img, img = process_img(input_img, long_size)
    ###
    print(org_img)
    cv2.namedWindow("org_img", cv2.WINDOW_NORMAL)
    cv2.imshow("org_img", org_img)
    cv2.waitKey(0)
    ###

    img = Variable(img.cuda(), volatile=True)
    #org_img = org_img.numpy().astype('uint8')[0]
    text_box = org_img.copy()

    outputs = model(img)
    # # Use torch.jit.trace to generate a torch.jit.ScriptModule via tracing.
    # traced_script_module = torch.jit.trace(model, img)
    # traced_script_module.save("./model.pt")

    score = torch.sigmoid(outputs[:, 0, :, :])
    outputs = (torch.sign(outputs - args.binary_th) + 1) / 2

    text = outputs[:, 0, :, :]
    kernels = outputs[:, 0:args.kernel_num, :, :] * text

    score = score.data.cpu().numpy()[0].astype(np.float32)
    text = text.data.cpu().numpy()[0].astype(np.uint8)
    kernels = kernels.data.cpu().numpy()[0].astype(np.uint8)
    
    # c++ version pse
    pred = pse(kernels, args.min_kernel_area / (args.scale * args.scale))
    # python version pse
    # pred = pypse(kernels, args.min_kernel_area / (args.scale * args.scale))
    
    # scale = (org_img.shape[0] * 1.0 / pred.shape[0], org_img.shape[1] * 1.0 / pred.shape[1])
    scale = (org_img.shape[1] * 1.0 / pred.shape[1], org_img.shape[0] * 1.0 / pred.shape[0])
    label = pred
    label_num = np.max(label) + 1
    bboxes = []
    for i in range(1, label_num):
        points = np.array(np.where(label == i)).transpose((1, 0))[:, ::-1]

        if points.shape[0] < args.min_area / (args.scale * args.scale):
            continue

        score_i = np.mean(score[label == i])
        if score_i < args.min_score:
            continue

        # rect = cv2.minAreaRect(points)
        binary = np.zeros(label.shape, dtype='uint8')
        binary[label == i] = 1

        _, contours, _ = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contour = contours[0]
        # epsilon = 0.01 * cv2.arcLength(contour, True)
        # bbox = cv2.approxPolyDP(contour, epsilon, True)
        bbox = contour

        if bbox.shape[0] <= 2:
            continue
        
        bbox = bbox * scale
        bbox = bbox.astype('int32')
        bboxes.append(bbox.reshape(-1))

    det_rects = []
    for bbox in bboxes:
        ###insert by yuyang###
        rect = cv2.minAreaRect(np.array([bbox.reshape(bbox.shape[0] / 2, 2)]))
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        r = cv2.boundingRect(np.array([bbox.reshape(bbox.shape[0] / 2, 2)]))
        det_rects.append(r)
    return det_rects

def general_text_detect_1(img_path, model, long_size = 1280):
    data_loader = OCRTestGeneralTextDetLoader_2(img_path, long_size=args.long_size)
    test_loader = torch.utils.data.DataLoader(
        data_loader,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        drop_last=True) #num_workers=2

    det_rects = []
    for idx, (org_img, img) in enumerate(test_loader):
        print("idx: {0}".format(idx))
        #img = Variable(img.cuda(), volatile=True)
        img = Variable(img, volatile=True)
        org_img = org_img.numpy().astype('uint8')[0]
        text_box = org_img.copy()
        outputs = model(img)
        # # Use torch.jit.trace to generate a torch.jit.ScriptModule via tracing.
        # traced_script_module = torch.jit.trace(model, img)
        # traced_script_module.save("./model.pt")

        score = torch.sigmoid(outputs[:, 0, :, :])
        outputs = (torch.sign(outputs - args.binary_th) + 1) / 2

        text = outputs[:, 0, :, :]
        kernels = outputs[:, 0:args.kernel_num, :, :] * text

        score = score.data.cpu().numpy()[0].astype(np.float32)
        text = text.data.cpu().numpy()[0].astype(np.uint8)
        kernels = kernels.data.cpu().numpy()[0].astype(np.uint8)
        
        # c++ version pse
        pred = pse(kernels, args.min_kernel_area / (args.scale * args.scale))
        # python version pse
        # pred = pypse(kernels, args.min_kernel_area / (args.scale * args.scale))
        
        # scale = (org_img.shape[0] * 1.0 / pred.shape[0], org_img.shape[1] * 1.0 / pred.shape[1])
        scale = (org_img.shape[1] * 1.0 / pred.shape[1], org_img.shape[0] * 1.0 / pred.shape[0])
        label = pred
        label_num = np.max(label) + 1
        bboxes = []
        for i in range(1, label_num):
            points = np.array(np.where(label == i)).transpose((1, 0))[:, ::-1]

            if points.shape[0] < args.min_area / (args.scale * args.scale):
                continue

            score_i = np.mean(score[label == i])
            if score_i < args.min_score:
                continue

            # rect = cv2.minAreaRect(points)
            binary = np.zeros(label.shape, dtype='uint8')
            binary[label == i] = 1

            _, contours, _ = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            contour = contours[0]
            # epsilon = 0.01 * cv2.arcLength(contour, True)
            # bbox = cv2.approxPolyDP(contour, epsilon, True)
            bbox = contour

            if bbox.shape[0] <= 2:
                continue
            
            bbox = bbox * scale
            bbox = bbox.astype('int32')
            bboxes.append(bbox.reshape(-1))

        
        for bbox in bboxes:
            ###insert by yuyang###
            rect = cv2.minAreaRect(np.array([bbox.reshape(bbox.shape[0] / 2, 2)]))
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            x, y, w, h = cv2.boundingRect(np.array([bbox.reshape(bbox.shape[0] / 2, 2)]))
            det_rects.append((x,y,w,h))
    return det_rects

def test(args):
    data_loader = OCRTestGeneralTextDetLoader_1(long_size=args.long_size)
    test_loader = torch.utils.data.DataLoader(
        data_loader,
        batch_size=1,
        shuffle=False,
        num_workers=2,
        drop_last=True)

    # Setup Model
    if args.arch == "resnet50":
        model = models.resnet50(pretrained=True, num_classes=7, scale=args.scale)
    elif args.arch == "resnet101":
        model = models.resnet101(pretrained=True, num_classes=7, scale=args.scale)
    elif args.arch == "resnet152":
        model = models.resnet152(pretrained=True, num_classes=7, scale=args.scale)
    
    for param in model.parameters():
        param.requires_grad = False

    model = model.cuda()
    
    if args.resume is not None:                                         
        if os.path.isfile(args.resume):
            print("Loading model and optimizer from checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            
            # model.load_state_dict(checkpoint['state_dict'])
            d = collections.OrderedDict()
            for key, value in checkpoint['state_dict'].items():
                tmp = key[7:]
                d[tmp] = value
            model.load_state_dict(d)

            print("Loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
            sys.stdout.flush()
        else:
            print("No checkpoint found at '{}'".format(args.resume))
            sys.stdout.flush()

    model.eval()
    
    total_frame = 0.0
    total_time = 0.0
    for idx, (org_img, img) in enumerate(test_loader):
        print('progress: %d / %d'%(idx, len(test_loader)))
        sys.stdout.flush()

        img = Variable(img.cuda(), volatile=True)
        org_img = org_img.numpy().astype('uint8')[0]
        text_box = org_img.copy()

        torch.cuda.synchronize()
        start = time.time()

        outputs = model(img)
        # # Use torch.jit.trace to generate a torch.jit.ScriptModule via tracing.
        # traced_script_module = torch.jit.trace(model, img)
        # traced_script_module.save("./model.pt")

        score = torch.sigmoid(outputs[:, 0, :, :])
        outputs = (torch.sign(outputs - args.binary_th) + 1) / 2

        text = outputs[:, 0, :, :]
        kernels = outputs[:, 0:args.kernel_num, :, :] * text

        score = score.data.cpu().numpy()[0].astype(np.float32)
        text = text.data.cpu().numpy()[0].astype(np.uint8)
        kernels = kernels.data.cpu().numpy()[0].astype(np.uint8)
        
        # c++ version pse
        pred = pse(kernels, args.min_kernel_area / (args.scale * args.scale))
        # python version pse
        # pred = pypse(kernels, args.min_kernel_area / (args.scale * args.scale))
        
        # scale = (org_img.shape[0] * 1.0 / pred.shape[0], org_img.shape[1] * 1.0 / pred.shape[1])
        scale = (org_img.shape[1] * 1.0 / pred.shape[1], org_img.shape[0] * 1.0 / pred.shape[0])
        label = pred
        label_num = np.max(label) + 1
        bboxes = []
        for i in range(1, label_num):
            points = np.array(np.where(label == i)).transpose((1, 0))[:, ::-1]

            if points.shape[0] < args.min_area / (args.scale * args.scale):
                continue

            score_i = np.mean(score[label == i])
            if score_i < args.min_score:
                continue

            # rect = cv2.minAreaRect(points)
            binary = np.zeros(label.shape, dtype='uint8')
            binary[label == i] = 1

            _, contours, _ = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            contour = contours[0]
            # epsilon = 0.01 * cv2.arcLength(contour, True)
            # bbox = cv2.approxPolyDP(contour, epsilon, True)
            bbox = contour

            if bbox.shape[0] <= 2:
                continue
            
            bbox = bbox * scale
            bbox = bbox.astype('int32')
            bboxes.append(bbox.reshape(-1))

        torch.cuda.synchronize()
        end = time.time()
        total_frame += 1
        total_time += (end - start)
        print('fps: %.2f'%(total_frame / total_time))
        sys.stdout.flush()

        ###insert by yuyang###
        text_box_minrect = text_box.copy()
        coordinates = {}
        coordinates["objects"] = []
        ######################

        for bbox in bboxes:
            cv2.drawContours(text_box, [bbox.reshape(bbox.shape[0] / 2, 2)], -1, (0, 255, 0), 2)
            ###insert by yuyang###
            rect = cv2.minAreaRect(np.array([bbox.reshape(bbox.shape[0] / 2, 2)]))
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            # for i in range(len(box)):
            #     print("box[{0}]: {1}, {2}".format(i, box[i], tuple(box[i])))
            for i in range(len(box)):
                cv2.line(text_box_minrect, tuple(box[i]), tuple(box[(i + 1) % len(box)]), (255, 255, 0), 1, cv2.LINE_AA)
            #cv2.line(text_box_minrect, tuple(box[i]), tuple(box[(i + 1) % len(box)]), (255, 255, 0), 1)
            #cv2.drawContours(text_box_minrect, [box], 0, (255, 255, 0), 1)
            tmp_dict = {}
            tmp_dict["label"] = "1"
            tmp_dict["polygon"] = box.tolist()
            #print("box:{0}".format(box))
            coordinates["objects"].append(tmp_dict)
            ######################


        image_name = data_loader.img_paths[idx].split('/')[-1].split('.')[0]
        write_result_as_txt(image_name, bboxes, 'outputs/submit_ocr/')
        
        #text_box = cv2.resize(text_box, (text.shape[1], text.shape[0]))
        debug(idx, data_loader.img_paths, [[text_box]], 'outputs/vis_ocr/')

        ###insert by yuyang###
        #text_box_minrect = cv2.resize(text_box_minrect, (text.shape[1], text.shape[0]))
        #画最小外接矩形框的结果图 以及生成预打标签的json文件
        debug(idx, data_loader.img_paths, [[text_box_minrect]], 'outputs/vis_ocr_minrect/')
        write_result_as_json(image_name, coordinates, 'outputs/res_json/')
        ######################

def get_img(img_path):
    try:
        img = cv2.imread(img_path)
        img = img[:, :, [2, 1, 0]]
    except Exception as e:
        print img_path
        raise
    return img

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Hyperparams')
    parser.add_argument('--arch', nargs='?', type=str, default='resnet50')
    parser.add_argument('--resume', nargs='?', type=str, default=None,    
                        help='Path to previous saved model to restart from')
    parser.add_argument('--binary_th', nargs='?', type=float, default=1.0,
                        help='Path to previous saved model to restart from')
    parser.add_argument('--kernel_num', nargs='?', type=int, default=3,
                        help='Path to previous saved model to restart from')
    parser.add_argument('--scale', nargs='?', type=int, default=1,
                        help='Path to previous saved model to restart from')
    parser.add_argument('--long_size', nargs='?', type=int, default=1280,
                        help='Path to previous saved model to restart from')
    parser.add_argument('--min_kernel_area', nargs='?', type=float, default=10.0,
                        help='min kernel area')
    parser.add_argument('--min_area', nargs='?', type=float, default=300.0,
                        help='min area')
    parser.add_argument('--min_score', nargs='?', type=float, default=0.93,
                        help='min score')
    
    args = parser.parse_args()
    #args.resume = "checkpoints/ocr_resnet50_bs_4_ep_200_2020-03-24_21_51_23_pretrain_ic17/checkpoint82.pth.tar"
    args.resume = "checkpoints/general_detection.tar"

    #test(args)
    img_path = "/data_1/data/general_text_det/test/text_image/2D0A9A23-08A7-9865-CFC8-B46F71214A08.jpg"
    input_img = cv2.imread(img_path)
    #input_img = get_img(img_path)
    save_flag = True
    save_root = "/data_2/result/通用模型定位结果/20200601"
    save_img_folder = os.path.join(save_root, "det_img")
    if not os.path.exists(save_img_folder):
        os.makedirs(save_img_folder)
    save_res_folder = os.path.join(save_root, "det_res")
    if not os.path.exists(save_res_folder):
        os.makedirs(save_res_folder)

    model = load_pse_model(args)
    #det_rects = general_text_detect(input_img, model)
    #det_rects = general_text_detect_1(input_img, model)
    file_path = "/data_1/data/general_text_det/test/text_image_413"
    #file_path = "/data_1/data/general_text_det/test/text_image_baobiao"
    total = 0
    max_time = 0
    i = 0
    for img_name in os.listdir(file_path):
        imgpath = os.path.join(file_path, img_name)
        inputImg = cv2.imread(imgpath)
        #torch.cuda.synchronize()
        start = time.time()
        det_rects = general_text_detect_1(inputImg, model, 640) #1280 640
        #torch.cuda.synchronize()
        end = time.time()
        print("per_img_time: {0}".format(end - start))
        total = total + (end - start)
        if end - start > max_time:
            max_time = end - start
        #det_rects = general_text_detect_1(imgpath, model)
        if save_flag:
            res_img = inputImg.copy()
            save_det_res_path = os.path.join(save_res_folder, "{0}.txt".format(os.path.splitext(img_name)[0]))
            with open(save_det_res_path, 'w') as fw:
                for (x, y, w, h) in det_rects:
                    cv2.rectangle(res_img, (x, y), (x + w, y + h), (255, 255, 0), 2)
                    fw.write("{0},{1},{2},{3}\n".format(x, y, x + w, y + h))

            save_img_path = os.path.join(save_img_folder, img_name)
            cv2.imwrite(save_img_path, res_img)
        i = i + 1
    
    print("i: {0}".format(i))
    print("avg_img_time: {0}".format(total / i))
    print("max_time: {0}".format(max_time))
    
    # cv2.namedWindow("img", cv2.WINDOW_NORMAL)
    # cv2.imshow("img", input_img)
    # cv2.waitKey(0)

