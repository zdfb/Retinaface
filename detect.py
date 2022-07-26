import cv2
import time
import torch
import argparse
import numpy as np
import torch.backends.cudnn as cudnn

from data import cfg_re152
from utils.load_model import load_model
from models.retinaface import RetinaFace
from utils.nms.py_cpu_nms import py_cpu_nms
from layers.functions.prior_box import PriorBox
from utils.box_utils import decode, decode_landm


parser = argparse.ArgumentParser(description = 'Retinaface')

parser.add_argument('-m', '--trained_model', default ='./weights/retinaface.pth', type = str, help = 'Trained state_dict file path to open')
parser.add_argument('--cpu', action = "store_true", default = False, help = 'Use cpu inference')

parser.add_argument('--confidence_threshold', default = 0.4, type = float, help = 'confidence_threshold')
parser.add_argument('--top_k', default = 5000, type = int, help='top_k')
parser.add_argument('--nms_threshold', default = 0.4, type = float, help = 'nms_threshold')
parser.add_argument('--keep_top_k', default=750, type=int, help='keep_top_k')
parser.add_argument('-s', '--save_image', action = "store_true", default = True, help = 'show detection results')
args = parser.parse_args()


def detect(image_path, net):

    img_raw = cv2.imread(image_path, cv2.IMREAD_COLOR)

    img = np.float32(img_raw)

    im_height, im_width, _ = img.shape
    scale = torch.Tensor([img.shape[1], img.shape[0], img.shape[1], img.shape[0]])
    img -= (104, 117, 123)
    img = img.transpose(2, 0, 1)
    img = torch.from_numpy(img).unsqueeze(0)
    img = img.to(device)
    scale = scale.to(device)

    tic = time.time()
    loc, conf, landms = net(img)  # forward pass
    print('net forward time: {:.4f}'.format(time.time() - tic))

    priorbox = PriorBox(cfg, image_size=(im_height, im_width))
    priors = priorbox.forward()
    priors = priors.to(device)
    prior_data = priors.data
    boxes = decode(loc.data.squeeze(0), prior_data, cfg['variance'])
    boxes = boxes * scale
    boxes = boxes.cpu().numpy()
    scores = conf.squeeze(0).data.cpu().numpy()[:, 1]
    landms = decode_landm(landms.data.squeeze(0), prior_data, cfg['variance'])
    scale1 = torch.Tensor([img.shape[3], img.shape[2], img.shape[3], img.shape[2],
                        img.shape[3], img.shape[2], img.shape[3], img.shape[2],
                        img.shape[3], img.shape[2]])
    scale1 = scale1.to(device)
    landms = landms * scale1
    landms = landms.cpu().numpy()

    # ignore low scores
    inds = np.where(scores > args.confidence_threshold)[0]
    boxes = boxes[inds]
    landms = landms[inds]
    scores = scores[inds]

    # keep top-K before NMS
    order = scores.argsort()[::-1][:args.top_k]
    boxes = boxes[order]
    landms = landms[order]
    scores = scores[order]

    # do NMS
    dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
    keep = py_cpu_nms(dets, args.nms_threshold)
    # keep = nms(dets, args.nms_threshold,force_cpu=args.cpu)
    dets = dets[keep, :]
    landms = landms[keep]

    # keep top-K faster NMS
    dets = dets[:args.keep_top_k, :]
    landms = landms[:args.keep_top_k, :]

    dets = np.concatenate((dets, landms), axis=1)

    return dets, img_raw


def save_image(dets, img_raw, name):
    for b in dets:
        text = "{:.4f}".format(b[4])
        b = list(map(int, b))
        
        cv2.rectangle(img_raw, (b[0], b[1]), (b[2], b[3]), (0, 0, 255), 2)

        cx = b[0]
        cy = b[1] + 12
        cv2.putText(img_raw, text, (cx, cy), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255))

        cv2.circle(img_raw, (b[5], b[6]), 1, (0, 0, 255), 4)
        cv2.circle(img_raw, (b[7], b[8]), 1, (0, 255, 255), 4)
        cv2.circle(img_raw, (b[9], b[10]), 1, (255, 0, 255), 4)
        cv2.circle(img_raw, (b[11], b[12]), 1, (0, 255, 0), 4)
        cv2.circle(img_raw, (b[13], b[14]), 1, (255, 0, 0), 4)

    cv2.imwrite(name, img_raw)


if __name__ == '__main__':
    torch.set_grad_enabled(False)
    cudnn.benchmark = True
    device = torch.device("cpu" if args.cpu else "cuda")

    cfg = cfg_re152
    net = RetinaFace(cfg = cfg, phase = 'test')
    net = load_model(net, args.trained_model, args.cpu)
    print('Finished loading model!')
    net.eval()
    net = net.to(device)

    image_path = "test_samples/selfie.jpg"
    dets, img_raw = detect(image_path, net)

    if args.save_image:
        save_image(dets, img_raw, 'result.jpg')
    

