import os
import random
import copy

import cv2
import numpy as np
import torch

from .module.prior_box import PriorBox
from .module.py_cpu_nms import py_cpu_nms
from .module.box_utils import decode


__all__ = ['OVMWorker']

CFG = {
    'min_sizes': [[8, 11], [14, 19, 26, 38, 64, 149]], 
    'steps': [8, 16],
    'variance': [0.1, 0.2],
    'clip': False,
}

class InferenceWorker:
    def __init__(self, frame_height, frame_width, device='cpu', nickname=""):        
        self.nickname = nickname
        self.device = torch.device(device)
        self.bgr_mean = np.array([104, 117, 123], dtype=np.float32)

        self.frame_height = frame_height
        self.frame_width = frame_width
        self.image_size_hw = (frame_height, frame_width)
        self.image_size_wh = (frame_width, frame_height)
        self.priorbox = PriorBox(CFG, image_size=self.image_size_hw)
        self.priors = self.priorbox.forward().to(self.device)
        self.cfg_variance = CFG['variance']

    def reinit(self, frame_height, frame_width): 
        self.frame_height = frame_height
        self.frame_width = frame_width
        self.image_size_hw = (frame_height, frame_width)
        self.image_size_wh = (frame_width, frame_height)
        self.priorbox = PriorBox(CFG, image_size=self.image_size_hw)
        self.priors = self.priorbox.forward().to(self.device)

    def preprocess(self, image, interp_method=cv2.INTER_AREA):
        """
        Preprocess image for inference.
        
        Args:
            image: Input BGR image
            interp_method: Interpolation method for resizing
            
        Returns:
            Preprocessed image tensors and metadata
        """
        # Original image dimensions
        h, w, _ = image.shape
        
        # Target dimensions
        target_h, target_w = self.image_size_hw
        
        # Calculate new size while maintaining aspect ratio
        scale = min(target_w / w, target_h / h)
        new_w, new_h = int(w * scale), int(h * scale)
                
        # Resize image        
        if new_w!=w or new_h!=h:
            resized_image = cv2.resize(image, (new_w, new_h), interpolation=interp_method)
        else:
            resized_image = copy.deepcopy(image)
        
        # Calculate padding to center the image
        delta_w = target_w - new_w
        delta_h = target_h - new_h

        image_RGB = resized_image.astype(np.float32) - self.bgr_mean
        image_RGB = image_RGB.transpose(2, 0, 1) # ==> C H W
        # Apply zero-padding after normalization
        top, bottom = delta_h // 2, delta_h - (delta_h // 2)
        left, right = delta_w // 2, delta_w - (delta_w // 2)            

        if delta_w!=0 or delta_h!=0:        
            image_RGB = np.pad(image_RGB, ((0, 0), (top, bottom), (left, right)), mode='constant', constant_values=0)
        return image_RGB, resized_image, [scale, top, left, h, w]

    def infer(self, image):
        raise NotImplementedError("This method should be implemented by subclasses.")

    def postprocess(self, loc, conf, 
        confidence_threshold=0.38, 
        nms_threshold=0.4,
        scale_top_left_height_width=None,
        ):
        prior_data = self.priors.data

        loc = [_.permute(0, 2, 3, 1).contiguous() for _ in loc]
        conf = [_.permute(0, 2, 3, 1).contiguous() for _ in conf]
        
        bbox_regressions = torch.cat([o.view(o.size(0), -1, 4) for o in loc], 1)

        RATIO = list(conf[0].shape)[-1] // len(CFG['min_sizes'][0])
        
        if RATIO==1:
            classifications = torch.cat([o.view(o.size(0), -1, 1) for o in conf], 1)    
        elif RATIO==2:        
            classifications = torch.cat([o.view(o.size(0), -1, 2) for o in conf], 1)    
            classifications = classifications[:,:,1]-classifications[:,:,0]
            classifications = classifications.unsqueeze(-1)
        else:
            raise ValueError('RATIO is expected to be 1 or 2')
                
        loc, conf = (bbox_regressions, torch.sigmoid(classifications))                    

        boxes = decode(loc.data.squeeze(0), prior_data, self.cfg_variance)

        boxes = boxes.cpu().numpy()
        
        boxes = np.clip(boxes, 0.0, 1.0)
        
        W, H = self.image_size_wh
        boxes[:,0] *= W
        boxes[:,2] *= W
        boxes[:,1] *= H
        boxes[:,3] *= H

        scores = conf.squeeze(0).data.cpu().numpy().squeeze(1)

        # ignore low scores
        inds = np.where(scores > confidence_threshold)[0]
        boxes = boxes[inds]    
        scores = scores[inds]

        # sort before NMS
        order = scores.argsort()[::-1]
        boxes = boxes[order]
        scores = scores[order]
        
        scale, top, left, height, width = scale_top_left_height_width
        # Adjust boxes to compensate for padding and scale difference
        boxes[:, [0, 2]] = (boxes[:, [0, 2]] - left) / scale
        boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0.0, width)

        boxes[:, [1, 3]] = (boxes[:, [1, 3]] - top) / scale
        boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0.0, height)

        # do NMS
        dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
        keep = py_cpu_nms(dets, nms_threshold)
        dets = dets[keep, :]        
        return dets

    def preprocess_infer_postprocess(self, frame_BGR, confidence_threshold, nms_threshold):
        h, w = frame_BGR.shape[:2]
        preprocessed_frame, image_padded_copy, scale_top_left_height_width = self.preprocess(frame_BGR)
        loc, conf = self.infer(preprocessed_frame)

        detections = self.postprocess(loc, conf, 
            scale_top_left_height_width=scale_top_left_height_width,
            confidence_threshold=confidence_threshold,
            nms_threshold=nms_threshold,
        )
        return image_padded_copy, detections

class OVMWorker(InferenceWorker):
    def __init__(self, model_path, frame_height, frame_width, device='cpu', nickname=""):
        super().__init__(frame_height, frame_width, device, nickname)
        from quartz.ovq import sim
        self.model = sim.OvmModule(ovm_fname=model_path)
        self.model.eval()
    def infer(self, image):
        tensor_image = torch.from_numpy(image).unsqueeze(0)
        ovm_outputs = self.model(tensor_image)

        bbox_preds = ovm_outputs[:2]
        cls_scores = ovm_outputs[2:]
        return bbox_preds, cls_scores
