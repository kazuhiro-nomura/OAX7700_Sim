import os
import cv2
import argparse
from pathlib import Path
from pylib import OVMWorker


def parse_args():
    parser = argparse.ArgumentParser(description="Run inference with OVMWorker based on quartz.")
    parser.add_argument("--model", type=str, default="176x304.c0",
                        choices=["160x288.c0", "176x304.c0", "176x304.c1", "256x320.c1", "208x256.c1", "208x256.c1q2"],
                        help="Model specifier, e.g. '176x304.c0'")
    parser.add_argument("--input_list", type=str, default="filelist/images_176x304.example.txt",
                        help="Path of input image list file")
    parser.add_argument("--output_dir", type=str, default="./outputs_pyinfer",
                        help="Path of output directory")
    parser.add_argument("--conf_thresh", type=str, default=0.38,
                        help="Thresh of confidence")
    parser.add_argument("--nms_thresh", type=str, default=0.4,
                        help="Thresh of IOU for NMS")
    return parser.parse_args()

def save_detections_to_txt(detections, txtfile, imgfile):
    with open(txtfile, 'w') as txtf:
        txtf.write(imgfile + "\n")
        txtf.write("confidence,x1,y1,x2,y2\n")
        for det in detections:
            txtf.write("{:.4f},{:.2f},{:.2f},{:.2f},{:.2f}\n".format(det[4], det[0], det[1], det[2], det[3]))

def main():
    args = parse_args()

    model = args.model
    input_list = args.input_list
    output_dir = args.output_dir
    confidence_threshold = float(args.conf_thresh)
    nms_threshold = float(args.nms_thresh)

    hxw, ver_ckpt = model.split('.')
    frame_height, frame_width = [int(_) for _ in hxw.split('x')]

    script_dir = Path(__file__).parent
    model_path = str(script_dir.parent / f"ovmfile/{frame_height}x{frame_width}/0204_{ver_ckpt}.xnpu.WTF_MAC_2K.ovm")
    os.makedirs(output_dir, exist_ok=True)

    worker = OVMWorker(model_path, frame_height, frame_width, device='cpu', nickname=model_path)

    list_imgfiles = [_.strip() for _ in open(input_list, 'r').readlines()]
    for idx_img, imgfile in enumerate(list_imgfiles):
        print(f'Processing {imgfile}')
        frame_BGR = cv2.imread(imgfile)
        image_padded_copy, detections = worker.preprocess_infer_postprocess(
            frame_BGR, confidence_threshold, nms_threshold
        )
        for idx_det, det in enumerate(detections):
            print("\tDetection {}: [x1={:.2f}, y1={:.2f}, x2={:.2f}, y2={:.2f}] Conf: {:.4f}".format(
                idx_det, det[0], det[1], det[2], det[3], det[4]))
        txtfile = os.path.join(output_dir, f"detections_{idx_img:04d}.txt")
        print(f'\t==> {txtfile}')
        save_detections_to_txt(detections, txtfile, imgfile)

if __name__ == "__main__":
    main()
