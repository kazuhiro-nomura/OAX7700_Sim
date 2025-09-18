import os
import cv2
import glob

def parse_detection_file(filepath):
    with open(filepath, 'r') as f:
        lines = f.read().strip().split('\n')
        image_path = lines[0]
        detections = []
        for line in lines[2:]:  # skip header
            parts = list(map(float, line.strip().split(',')))
            detections.append({
                'confidence': parts[0],
                'x1': parts[1],
                'y1': parts[2],
                'x2': parts[3],
                'y2': parts[4]
            })
        return image_path, detections

def draw_detections(image, detections, 
    color_highconf=(51, 255, 51), 
    color_lowconf=(255, 128, 0), 
    thre_conf_high=0.45
    ):
    for det in detections:
        x1, y1, x2, y2 = map(int, [det['x1'], det['y1'], det['x2'], det['y2']])
        conf = det['confidence']
        color_det = color_highconf if conf>thre_conf_high else color_lowconf
        cv2.rectangle(image, (x1, y1), (x2, y2), color_det, 2)
        cv2.putText(image, f"{conf:.2f}", (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, 
                    color_det, 1)
    return image

def main(root_output='outputs'):
    for model_choice in ["160x288", "176x304"]:
        output_dir = os.path.join(root_output, model_choice)
        if os.path.exists(output_dir):
            output_viz_dir = os.path.join(f'{root_output}_viz', model_choice)
            os.makedirs(output_viz_dir, exist_ok=True)
            detection_files = sorted(glob.glob(os.path.join(output_dir, 'detections_*.txt')))

            for det_file in detection_files:
                image_path, detections = parse_detection_file(det_file)
                if not os.path.exists(image_path):
                    print(f"Image not found: {image_path}")
                    continue

                img = cv2.imread(image_path)
                if img is None:
                    print(f"Failed to load image: {image_path}")
                    continue

                img_with_dets = draw_detections(img, detections)
                img_file = os.path.join(output_viz_dir, os.path.basename(det_file) + '.jpg')
                print(f'Wrting to {img_file}')
                cv2.imwrite(img_file, img_with_dets)

if __name__ == '__main__':
    main(root_output='outputs')
