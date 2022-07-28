import os
import cv2
import torch
import argparse
import numpy as np
from data import  cfg_re152
import torch.backends.cudnn as cudnn

from utils.load_model import load_model
from models.retinaface import RetinaFace
from utils.nms.py_cpu_nms import py_cpu_nms
from layers.functions.prior_box import PriorBox
from utils.box_utils import decode, decode_landm


parser = argparse.ArgumentParser(description = 'Retinaface')
parser.add_argument('-m', '--trained_model', default = './weights/retinaface.pth', type = str, help = 'Trained state_dict file path to open')
parser.add_argument('--save_folder', default ='./widerface_evaluate/widerface_txt/', type = str, help = 'Dir to save txt results')
parser.add_argument('--cpu', action = "store_true", default = False, help = 'Use cpu inference')

parser.add_argument('--dataset_folder', default = './data/widerface/val/images/', type = str, help = 'dataset path')
parser.add_argument('--confidence_threshold', default = 0.02, type = float, help ='confidence_threshold')
parser.add_argument('--top_k', default = 5000, type = int, help = 'top_k')
parser.add_argument('--nms_threshold', default = 0.4, type = float, help = 'nms_threshold')
parser.add_argument('--keep_top_k', default = 750, type = int, help = 'keep_top_k')

parser.add_argument('-s', '--save_image', action = "store_true", default = False, help = 'show detection results')
parser.add_argument('--vis_thres', default = 0.5, type = float, help = 'visualization_threshold')
args = parser.parse_args()


def detect_face(img, net, resize):
    img = cv2.resize(img, None, None, fx = resize, fy = resize, interpolation = cv2.INTER_LINEAR)

    im_height, im_width, _ = img.shape
    scale = torch.Tensor([img.shape[1], img.shape[0], img.shape[1], img.shape[0]])

    img -= (104, 117, 123)
    img = img.transpose(2, 0, 1)
    img = torch.from_numpy(img).unsqueeze(0)
    img = img.to(device)
    scale = scale.to(device)

    loc, conf, landms = net(img)  # forward pass
  
    priorbox = PriorBox(cfg, image_size = (im_height, im_width))
    priors = priorbox.forward()
    priors = priors.to(device)
    prior_data = priors.data
    boxes = decode(loc.data.squeeze(0), prior_data, cfg['variance'])
    boxes = boxes * scale / resize
    boxes = boxes.cpu().numpy()
    scores = conf.squeeze(0).data.cpu().numpy()[:, 1]
    landms = decode_landm(landms.data.squeeze(0), prior_data, cfg['variance'])

    scale1 = torch.Tensor([img.shape[3], img.shape[2], img.shape[3], img.shape[2],
                            img.shape[3], img.shape[2], img.shape[3], img.shape[2],
                            img.shape[3], img.shape[2]])
    scale1 = scale1.to(device)
    landms = landms * scale1 / resize
    landms = landms.cpu().numpy()

    # ignore low scores
    inds = np.where(scores > args.confidence_threshold)[0]
    boxes = boxes[inds]
    landms = landms[inds]
    scores = scores[inds]

    # keep top-K before NMS
    order = scores.argsort()[::-1]
    # order = scores.argsort()[::-1][:args.top_k]
    boxes = boxes[order]
    landms = landms[order]
    scores = scores[order]

    # do NMS
    dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
    keep = py_cpu_nms(dets, args.nms_threshold)
    # keep = nms(dets, args.nms_threshold,force_cpu=args.cpu)
    dets = dets[keep, :]
    landms = landms[keep, :]

    return dets, landms


def bbox_vote(dets, landms):
    keep = py_cpu_nms(dets, 0.2)
    dets = dets[keep, :]
    landms = landms[keep, :]

    return dets, landms


def flip_detect(img, net, resize):
    img_f = cv2.flip(img, 1)
    dets_f, landms_f = detect_face(img_f, net, resize)

    dets_t = np.zeros(dets_f.shape)
    landms_t = np.zeros(landms_f.shape)

    dets_t[:, 0] = img.shape[1] - dets_f[:, 2]
    dets_t[:, 1] = dets_f[:, 1]
    dets_t[:, 2] = img.shape[1] - dets_f[:, 0]
    dets_t[:, 3] = dets_f[:, 3]
    dets_t[:, 4] = dets_f[:, 4]

    landms_t[:, 0] = img.shape[1] - landms_f[:, 8]
    landms_t[:, 1] = landms_f[:, 1]
    landms_t[:, 2] = img.shape[1] - landms_f[:, 6]
    landms_t[:, 3] = landms_f[:, 3]
    landms_t[:, 4] = img.shape[1] - landms_f[:, 4]
    landms_t[:, 5] = landms_f[:, 5]
    landms_t[:, 6] = img.shape[1] - landms_f[:, 2]
    landms_t[:, 7] = landms_f[:, 7]
    landms_t[:, 8] = img.shape[1] - landms_f[:, 0]
    landms_t[:, 9] = landms_f[:, 9]

    return dets_t, landms_t

def multi_detect(img, net):
    dets_list = []
    landms_list = []

    dets, landms = detect_face(img, net, 1)
    dets_f, landms_f = flip_detect(img, net, 1)

    dets_list.append(dets)
    dets_list.append(dets_f)

    landms_list.append(landms)
    landms_list.append(landms_f)

    im_shape = img.shape
    im_size_min = np.min(im_shape[0:2])

    target_sizes = [500, 800, 1100, 1400, 1700]
    for target_size in target_sizes:
        resize = float(target_size) / float(im_size_min)
        dets, landms = detect_face(img, net, resize)
        dets_list.append(dets)
        landms_list.append(landms)
    
    multi_scale_dets = np.row_stack(dets_list)
    multi_scale_landms = np.row_stack(landms_list)

    dets, landms = bbox_vote(multi_scale_dets, multi_scale_landms)

    dets = np.concatenate((dets, landms), axis=1)
    return dets


if __name__ == '__main__':
    torch.set_grad_enabled(False)

    cfg = cfg_re152
    net = RetinaFace(cfg = cfg, phase = 'test')
    net = load_model(net, args.trained_model, args.cpu)
    net.eval()
    print('Finished loading model!')
    cudnn.benchmark = True
    device = torch.device("cpu" if args.cpu else "cuda")
    net = net.to(device)

    # testing dataset
    testset_folder = args.dataset_folder
    testset_list = args.dataset_folder[:-7] + "wider_val.txt"

    with open(testset_list, 'r') as fr:
        test_dataset = fr.read().split()
    num_images = len(test_dataset)


    # testing begin
    for i, img_name in enumerate(test_dataset):
        image_path = testset_folder + img_name
        img_raw = cv2.imread(image_path, cv2.IMREAD_COLOR)
        img = np.float32(img_raw)

        dets = multi_detect(img, net)
        # --------------------------------------------------------------------
        save_name = args.save_folder + img_name[:-4] + ".txt"
        dirname = os.path.dirname(save_name)
        if not os.path.isdir(dirname):
            os.makedirs(dirname)
        with open(save_name, "w") as fd:
            bboxs = dets
            file_name = os.path.basename(save_name)[:-4] + "\n"
            bboxs_num = str(len(bboxs)) + "\n"
            fd.write(file_name)
            fd.write(bboxs_num)
            for box in bboxs:
                x = int(box[0])
                y = int(box[1])
                w = int(box[2]) - int(box[0])
                h = int(box[3]) - int(box[1])
                confidence = str(box[4])
                line = str(x) + " " + str(y) + " " + str(w) + " " + str(h) + " " + confidence + " \n"
                fd.write(line)
        print('im_detect: {:d}/{:d}'.format(i + 1, num_images))

        # save image
        if args.save_image:
            for b in dets:
                if b[4] < args.vis_thres:
                    continue
                text = "{:.4f}".format(b[4])
                b = list(map(int, b))
                cv2.rectangle(img_raw, (b[0], b[1]), (b[2], b[3]), (0, 0, 255), 2)
                cx = b[0]
                cy = b[1] + 12
                cv2.putText(img_raw, text, (cx, cy),
                            cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255))

                # landms
                cv2.circle(img_raw, (b[5], b[6]), 1, (0, 0, 255), 4)
                cv2.circle(img_raw, (b[7], b[8]), 1, (0, 255, 255), 4)
                cv2.circle(img_raw, (b[9], b[10]), 1, (255, 0, 255), 4)
                cv2.circle(img_raw, (b[11], b[12]), 1, (0, 255, 0), 4)
                cv2.circle(img_raw, (b[13], b[14]), 1, (255, 0, 0), 4)
            # save image
            if not os.path.exists("./results/"):
                os.makedirs("./results/")
            name = "./results/" + str(i) + ".jpg"
            cv2.imwrite(name, img_raw)
