import os
import shutil
import subprocess
from pathlib import Path

import cv2
import json
import datetime
import numpy as np

from viz_detections import parse_detection_file, draw_detections

def module_process(infile, outfile, workdir="", cfg={}):
	w = cfg["in_width"]
	h = cfg["in_height"]

	incfgfile = infile + ".json"
	outcfgfile = outfile + ".json"

	if "force" in cfg and cfg["force"] == 0:
		if os.path.isfile(outcfgfile):
			print("[INFO] skip OD because output file exist")
			return outcfgfile

	# Clean workdir
	if workdir and os.path.exists(workdir):
		for item in os.listdir(workdir):
			item_path = os.path.join(workdir, item)
			if os.path.isdir(item_path):
				shutil.rmtree(item_path)
			else:
				os.remove(item_path)

	# Save frames
	rgb = np.fromfile(infile, dtype=np.uint8).reshape(-1, h, w, 3)
	for i in range(rgb.shape[0]):
		frame_path = os.path.join(workdir, f'ovtpd_FN_{i:04d}.png')
		cv2.imwrite(frame_path, cv2.cvtColor(rgb[i], cv2.COLOR_RGB2BGR))

	# Generate filelist
	filelist_path = os.path.join(workdir, 'filelist.txt')
	with open(filelist_path, 'w') as f:
		for png_file in sorted(Path(workdir).glob('*.png')):
			f.write(str(png_file.absolute()) + '\n')

	# Run the ovtpd python interface
	executable = Path(__file__).parent / "pyinfer.py"
	ovtpd_cmd = [
		'python',
		str(executable),
		'--model', cfg['param']['model'],
		'--input_list', filelist_path,
		'--output_dir', workdir,
		'--conf_thresh', str(cfg['param']['conf_thresh']),
		'--nms_thresh', str(cfg['param']['nms_thresh'])
	]
	subprocess.run(ovtpd_cmd, check=True)

	# Generate video result
	video_output = outfile + '.mp4'
	detection_files = sorted(Path(workdir).glob('detections_*.txt'))
	for det_file in detection_files:
		image_path, detections = parse_detection_file(det_file)
		img = cv2.imread(image_path)
		img_with_dets = draw_detections(img, detections)
		cv2.imwrite(image_path.replace('.png', '_vis.png'), img_with_dets)

	video_cmd = [
		cfg['ffmpeg_path'].rstrip(' -y'),
		'-y',
		'-framerate', '24',
		'-i', os.path.join(workdir, 'ovtpd_FN_%4d_vis.png'),
		'-c:v', 'libx264',
		'-pix_fmt', 'yuv420p',
		video_output
	]
	subprocess.run(video_cmd, check=True)

	# Generate JSON result
	if os.path.isfile(incfgfile):
		shutil.copyfile(incfgfile, outcfgfile)
		with open(outcfgfile) as f:
			outfdata = json.load(f)
		outfdata["results"] = {}
		outfdata["results"]["objects"] = ['person']
		for i, det_file in enumerate(detection_files):
			image_path, detections = parse_detection_file(det_file)
			for det in detections:
				outfdata["results"]["objects"].append({
					'frame': i,
					'confidence': round(det['confidence'], 2),
					'left': int(det['x1']),
					'top': int(det['y1']),
					'width': int((det['x2'] - det['x1'])),
					'height': int((det['y2'] - det['y1']))
				})
		outfdata["results"]["time"] = datetime.datetime.now().isoformat()
		
		with open(outcfgfile, "w") as f:
			json.dump(outfdata, f, indent=4)

	return outcfgfile
