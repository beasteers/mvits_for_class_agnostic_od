import os
import tqdm
import cv2
from PIL import Image

import numpy as np
import torch
from torch import nn
from models.model import _make_mdef_detr

import torchvision.transforms as T
from utils.nms import nms

torch.set_grad_enabled(False)




class ModulatedDetection(nn.Module):
    """
    The class supports the inference using both MDETR & MDef-DETR models.
    """
    def __init__(self, checkpoint, confidence_thresh=0.0):
        self.model = _make_mdef_detr(checkpoint).cuda()
        self.model.eval()
        self.conf_thresh = confidence_thresh
        self.transform = T.Compose([
            T.Resize(800),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def _prepare_image(self, im):
        if isinstance(im, np.ndarray):
            im = Image.fromarray(im[:,:,::-1])
        return self.transform(im).unsqueeze(0).cuda()

    def predict(self, im, caption, multi_crop=False, iou=0.5):
        if multi_crop:
            crops, coordinates, img_dims = generate_image_crops(im)
            ims = torch.cat([self._prepare_image(im) for im in crops])
        else:
            ims = self._prepare_image(im)

        # propagate through the models
        memory_cache = self.model(ims, [caption] * len(ims), encode_and_save=True)
        outputs = self.model(ims, [caption], encode_and_save=False, memory_cache=memory_cache)

        all_boxes, all_probs = [], []
        for i in range(len(ims)):
            # keep only predictions with self.conf_thresh+ confidence
            probs = 1 - outputs['pred_logits'].softmax(-1)[i, :, -1].cpu()
            keep = (probs > self.conf_thresh).cpu()
            # convert boxes from [0; 1] to image scales
            bboxes = rescale_bboxes(outputs['pred_boxes'].cpu()[i, keep], im.size)
            probs = probs[keep]
            if multi_crop:
                bboxes = rescale_crop(bboxes, coordinates[i], img_dims)
            all_boxes.append(bboxes.numpy())
            all_probs.append(probs.numpy())
        boxes = np.concatenate(all_boxes)
        probs = np.concatenate(all_probs)

        if multi_crop:
            boxes, probs = nms(boxes, probs, iou)
        return boxes, probs


def generate_image_crops(img, num_crops=8):
    """
    Note: num_crops must be greater than 2 and of multiple of 2
    """
    assert num_crops > 2 and num_crops % 2 == 0
    # Get the image width and height
    img_w, img_h = img.size
    crops = []
    coordinates = []
    crops.append(img)
    coordinates.append((0, 0, img_w, img_h))
    crop_chunks_x = int(num_crops / 2)
    crop_chunks_y = int(num_crops / crop_chunks_x)
    x_inc = int(img_w / crop_chunks_y)
    y_inc = int(img_h / crop_chunks_y)
    x_space = np.linspace(0, img_w - x_inc, crop_chunks_y)
    y_spcae = np.linspace(0, img_h - y_inc, int(num_crops / crop_chunks_y))
    if num_crops > 1:
        for x in x_space:
            for y in y_spcae:
                x1, y1 = x, y
                x2, y2 = x1 + x_inc, y1 + y_inc
                crops.append((img.crop((x1, y1, x2, y2))).resize((img_w, img_h)))
                coordinates.append((x1, y1, x2, y2))
    return crops, coordinates, (img_w, img_h)

def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(1)
    return torch.stack([
        (x_c - 0.5 * w), (y_c - 0.5 * h),
        (x_c + 0.5 * w), (y_c + 0.5 * h)], dim=1)

def rescale_bboxes(out_bbox, size):
    img_w, img_h = size
    return box_cxcywh_to_xyxy(out_bbox) * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32)

def rescale_crop(boxes, coordinates, img_dims):
    x1, y1, x2, y2 = coordinates
    img_w, img_h = img_dims
    boxes[0] = boxes[0] / img_w * (x2 - x1) + x1
    boxes[1] = boxes[1] / img_h * (y2 - y1) + y1
    boxes[2] = boxes[2] / img_w * (x2 - x1) + x1
    boxes[3] = boxes[3] / img_h * (y2 - y1) + y1
    return boxes








def _video_feed(src=0, fps=None):
    cap = cv2.VideoCapture(src)
    src_fps = cap.get(cv2.CAP_PROP_FPS)
    every = int(src_fps/fps) if fps else 1
    i = 0
    total = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    pbar = tqdm.tqdm(total=total)
    while True:
        ret, im = cap.read()
        i += 1
        pbar.update()
        if not ret:
            break
        if i%every: 
            continue
        yield (i-1) / fps, im



class ImageOutput:
    def __init__(self, src, fps, cc='avc1', show=None):
        self.src = src
        self.cc = cc
        self.fps = fps
        self._show = not src if show is None else show

    def __enter__(self):
        return self

    def __exit__(self, *a):
        if self._w:
            self._w.release()
        self._w = None
        if self._show:
            cv2.destroyAllWindows()

    def output(self, im):
        if self.src:
            self.write_video(im)
        if self._show:
            self.show_video(im)

    _w = None
    def write_video(self, im):
        if not self._w:
            self._w = cv2.VideoWriter(
                self.src, cv2.VideoWriter_fourcc(*self.cc),
                self.fps, im.shape[:2][::-1], True)
        self._w.write(im)

    def show_video(self, im):
        cv2.imshow('output', im)
        if cv2.waitKey(1) == ord('q'):  # q to quit
            raise StopIteration



localfile = lambda *fs: os.path.join(os.path.dirname(__file__), *fs)
CHECKPOINT = localfile('models/epoch=2-step=99021.ckpt')

def main(caption, src=0, out_file=None, checkpoint=CHECKPOINT, multi_crop=False, fps=10, show=None):
    model = ModulatedDetection(checkpoint)

    with ImageOutput(out_file, fps, show=show) as imout:
        for t, im in _video_feed(src, fps):
            bboxes, probs = model.predict(im, caption, multi_crop=multi_crop)
            for label, xyxy in zip(probs, bboxes):
                im = cv2.rectangle(im, xyxy[:2], xyxy[2:], (0,255,0), 2)
                im = cv2.putText(im, str(label), xyxy[:2], cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            imout.output(im)

if __name__ == '__main__':
    import fire
    fire.Fire(main)