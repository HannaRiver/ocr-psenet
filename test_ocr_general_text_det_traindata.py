import os
import cv2
import sys
import time
import collections
import torch
import argparse
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable
from torch.utils import data

from dataset import OCRTestGeneralTextDetLoader
from metrics import runningScore
import models
import util
# c++ version pse based on opencv 3+
from pse import pse
# python pse
# from pypse import pse as pypse

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

def cal_text_score(texts, gt_texts, training_masks, running_metric_text):
    training_masks = training_masks.data.cpu().numpy()
    pred_text = torch.sigmoid(texts).data.cpu().numpy() * training_masks
    pred_text[pred_text <= 0.5] = 0
    pred_text[pred_text >  0.5] = 1
    pred_text = pred_text.astype(np.int32)
    gt_text = gt_texts.data.cpu().numpy() * training_masks
    gt_text = gt_text.astype(np.int32)
    running_metric_text.update(gt_text, pred_text)
    score_text, _ = running_metric_text.get_scores()
    return score_text

def cal_kernel_score(kernels, gt_kernels, gt_texts, training_masks, running_metric_kernel):
    mask = (gt_texts * training_masks).data.cpu().numpy()
    kernel = kernels[:, -1, :, :]
    gt_kernel = gt_kernels[:, -1, :, :]
    pred_kernel = torch.sigmoid(kernel).data.cpu().numpy()
    pred_kernel[pred_kernel <= 0.5] = 0
    pred_kernel[pred_kernel >  0.5] = 1
    pred_kernel = (pred_kernel * mask).astype(np.int32)
    gt_kernel = gt_kernel.data.cpu().numpy()
    gt_kernel = (gt_kernel * mask).astype(np.int32)
    running_metric_kernel.update(gt_kernel, pred_kernel)
    score_kernel, _ = running_metric_kernel.get_scores()
    return score_kernel

def test(args):
    data_loader = OCRTestGeneralTextDetLoader(long_size=args.long_size)
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

    running_metric_text = runningScore(2)
    running_metric_kernel = runningScore(2)

    #for idx, (org_img, img) in enumerate(test_loader):
    for idx, (org_img, img, gt_texts, gt_kernels, testing_masks) in enumerate(test_loader):
        print('progress: %d / %d'%(idx, len(test_loader)))
        sys.stdout.flush()

        img = Variable(img.cuda(), volatile=True)
        org_img = org_img.numpy().astype('uint8')[0]
        text_box = org_img.copy()
        gt_texts = Variable(gt_texts.cuda())
        gt_kernels = Variable(gt_kernels.cuda())
        testing_masks = Variable(testing_masks.cuda())

        torch.cuda.synchronize()
        start = time.time()

        outputs = model(img)
        texts = outputs[:, 0, :, :]
        kernels = outputs[:, 1:, :, :]

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

        for bbox in bboxes:
            cv2.drawContours(text_box, [bbox.reshape(bbox.shape[0] / 2, 2)], -1, (0, 255, 0), 2)

        image_name = data_loader.img_paths[idx].split('/')[-1].split('.')[0]
        write_result_as_txt(image_name, bboxes, 'outputs/submit_ocr/')
        
        text_box = cv2.resize(text_box, (text.shape[1], text.shape[0]))
        debug(idx, data_loader.img_paths, [[text_box]], 'outputs/vis_ocr/')

        score_text = cal_text_score(texts, gt_texts, testing_masks, running_metric_text)
        #score_kernel = cal_kernel_score(kernels, gt_kernels, gt_texts, testing_masks, running_metric_kernel)
    
    #return score_text['Mean Acc'], score_kernel['Mean Acc'], score_text['Mean IoU'], score_kernel['Mean IoU']
    return score_text['Mean Acc'], score_text['Mean IoU']

if __name__ == '__main__':
    flag = "single_test"

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
    if "single_test" == flag:
        args.resume = "checkpoints/ocr_resnet50_bs_4_ep_60_2020-01-13_09_18_42_pretrain_ic17/checkpoint123.pth.tar"
        test_te_acc, test_te_iou = test(args)
        with open("/data_1/software/PseNet_1/outputs/acc.txt", 'w') as fw:
            fw.write("Test Acc: {0}, Test IOU: {1}".format(test_te_acc, test_te_iou))
    elif "batch_test" == flag:
        with open("/data_1/software/PseNet_1/outputs/batch_acc.txt", 'w') as fres:
            fres.write("Acc.        IOU.\n")
            best_acc = 0.0 
            best_iou = 0.0 
            best_model = ""
            with open("/data_1/software/PseNet_1/outputs/model_list.txt") as fr:
                for model_path in fr.readlines():
                    model_path = model_path.rstrip()
                    args.resume = model_path
                    test_te_acc, test_te_iou = test(args)
                    fres.write("{0}        {1}\n".format(test_te_acc, test_te_iou))
                    if test_te_acc > best_acc:
                        best_acc = test_te_acc
                        best_iou = test_te_iou
                        best_model = model_path
            fres.write("The best model is {0}, best acc is {1}, best iou is {2}".format(best_model, best_acc, best_iou))  
        
