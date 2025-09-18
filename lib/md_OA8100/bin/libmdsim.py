#!/usr/bin/python3

import os
import re
import json
import sys
import platform
import shutil
import glob
import datetime
import numpy as np
import cv2

def sorted_alphanumeric(data):
	convert = lambda text: int(text) if text.isdigit() else text.lower()
	alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ]
	return sorted(data, key=alphanum_key)

def libmd_process(args, chip):
	pydir=os.path.dirname(os.path.abspath(__file__))
	if platform.system() == "Windows":
		md_exe=os.path.join(pydir, "win", "md_sim_{0}.exe".format(chip))
	else:
		md_exe=os.path.join(pydir, "linux", "md_sim_{0}".format(chip))

	for arg in args:
		md_exe += " {0}".format(arg.replace(' ', '\\ '))

	print(md_exe)
	os.system(md_exe)

def libmd_process_fromargs(argv, chip):
	libmd_process(argv[1:], chip)

def create_polygon_matrix(in_w, in_h, polygon_points):
    """
    create a matrix with width x height, and fill the polygon area with 1.
    Args:
        width: the width of the matrix
        height: the height of the matrix
        polygon_points: the points list of the polygon, format as [(x1, y1), (x2, y2), ...]

    Returns:
        a 2D numpy array, with 1 in the polygon area, and 0 in the rest area.
    """
    matrix = np.zeros((in_h, in_w), dtype=np.uint8)

    #transform polygon_points to numpy array
    pts = np.array([polygon_points], dtype=np.int32)
    cv2.fillPoly(matrix, pts, color=1)

    return matrix

def downsample_matrix(matrix, block_size=8, method= 1):
    """
    downsample a matrix by block_size x block_size.
    Args:
        matrix: the input matrix
        block_size: the block size to downsample
        method: the downsample method, 1 for block_size x block_size block, 2 for center point block
    Returns:
        the downsampled matrix.
    """
    h, w = matrix.shape
    # the size of the downsampled matrix/block number
    down_h = (h - block_size) // block_size
    down_w = (w - block_size) // block_size

    if method == 1:
        # each block_size x block_size block has at least one 1, then the corresponding point in the downsampled matrix is 1.
        # use strided block to calculate
        temp_down_h  = h // block_size
        temp_down_w = w // block_size
        downsampled = matrix.reshape(temp_down_h, block_size, temp_down_w, block_size).max(axis=(1, 3))
        if temp_down_h != down_h or temp_down_w != down_w:
            print(f"temp_down_h: {temp_down_h}, temp_down_w: {temp_down_w}, down_h: {down_h}, down_w: {down_w}")
            downsampled = downsampled[:down_h, :down_w]
    else:
        # calculate the center point of each block_size x block_size block, downsample logic: only when the center point of the block is 1, the corresponding point in the downsampled matrix is 1
        downsampled = np.zeros((down_h, down_w), dtype=np.uint8)
        for i in range(down_h):
            for j in range(down_w):
                # calculate the center point of the current block
                center_y = i * block_size + block_size // 2
                center_x = j * block_size + block_size // 2
                if matrix[center_y, center_x] == 1:
                    downsampled[i, j] = 1

    return downsampled

def generate_dz_bitmap(cfg, workdir, outfile):
	with open(cfg) as f:
		fdata=json.load(f)
		if "distzone" in fdata:
			for k in fdata["distzone"]:
				dist_pts = fdata["distzone"][k]
				if len(dist_pts) > 2:
					#generate dz_bitmap and output to wordir
					matrix = create_polygon_matrix(fdata["width"], fdata["height"], dist_pts)
					downsampled = downsample_matrix(matrix, block_size=8, method=2)

					bitmask = np.packbits(downsampled.astype(np.uint8))
					output_bitmask_path = os.path.join(workdir, 'dz_bitmap_' + k + '.bin')
					with open(output_bitmask_path, 'wb') as f:
						bitmask.tofile(f)
					#copy to output dir
					outfile_path = os.path.join(outfile, 'dz_bitmap_' + k + '.bin')
					os.makedirs(os.path.dirname(outfile_path), exist_ok = True)
					shutil.copy(output_bitmask_path, outfile_path)
				else:
					print("ERROR: {0} distzone list should have at least 3 points to define a polygon".format(k))

def libmd_module_process(chip, infile, outfile, workdir="", cfg={}):
	param=cfg["param"]
	if "p" not in param:
		param["p"] = {}
	param["p"]["InputFilePath"] = infile
	param["p"]["OutputFilePath"] = workdir

	incfgfile=infile + ".json"
	outcfgfile=outfile + ".json"

	if "force" in cfg and cfg["force"] == 0:
		if os.path.isfile(outcfgfile):
			print("[INFO] skip MD because output file exist")
			return outcfgfile

	argv=[]
	for fkey in param:
		if fkey == "p":
			for pk in param["p"]:
				argv.append("-p")
				argv.append("{0}={1}".format(pk, param["p"][pk]))
		else:
			argv.append("-{0}".format(fkey))
			argv.append("{0}".format(param[fkey]))

	libmd_process(argv, chip)

	# generate dz_bitmap if needed
	if "out_dz_bitmap" in cfg.keys():
		if cfg["out_dz_bitmap"] == 1:
			if os.path.isfile(incfgfile):
				generate_dz_bitmap(incfgfile, workdir, outfile)

	# generate the default.cfg file in workdir, for MD tuning tool open
	with open(param["d"]) as f:
		outyuvname=os.path.join(workdir, os.path.basename(infile))
		shutil.copyfile(infile, outyuvname)
		outcfgname=os.path.join(workdir, "default.cfg")
		print(f"generate default.cfg file: {outcfgname}")
		fout = open(outcfgname, "w")
		while True:
			s = f.readline()
			if s=="":
				break
			if s.startswith("InputFilePath"):
				continue
			if s.startswith("OutputFilePath"):
				continue
			if s.startswith("InputWidth"):
				continue
			if s.startswith("InputHeight"):
				continue
			if s.startswith("InputDist0BitmapFilepath"):
				continue
			if s.startswith("InputDist1BitmapFilepath"):
				continue
			fout.write(s)
		#Note: MD default.cfg requires the format is 'parameter = value', need to add space after parameter name and before value,
		#or else the md_tuning tool will report error when using 'update' fucntion to rerun the MD online
		fout.write('\n') # If there is no '\n' at the end of the last line of input config file (param["d"]), in the output config file (default.cfg), the 'InputFilePath' item will be at the same line as the last item of param["d"]. Add '\n' to fix this issue.
		fout.write('InputFilePath = "{0}"\n'.format(os.path.basename(infile)))
		fout.write('OutputFilePath = "."\n') #don't use default "", or else MD uses it will output '/prre_xx.yuv', please use 'output' or '.' instead
		fout.write('InputWidth = {0}\n'.format(cfg["in_width"]))
		fout.write('InputHeight = {0}\n'.format(cfg["in_height"]))
		#todo: add bit map file path to default.cfg	aims for MD tuning tool	usage
		if(os.path.isfile(os.path.join(workdir, 'dz_bitmap_0.5m.bin'))):
			fout.write('InputDist0BitmapFilepath = "dz_bitmap_0.5m.bin"\n')
		if(os.path.isfile(os.path.join(workdir, 'dz_bitmap_1m.bin'))):
			fout.write('InputDist1BitmapFilepath = "dz_bitmap_1m.bin"\n')
		fout.close()

	# update json file, just copy
	if os.path.isfile(incfgfile):
		shutil.copyfile(incfgfile, outcfgfile)
		with open(outcfgfile) as f:
			outfdata=json.load(f)
		outfdata["results"] = {}
		outfdata["results"]["objects"] = []
		ismotion=0
		# TODO: append MD's output result to outcfgfile["results"]
		for file in sorted_alphanumeric(glob.glob(os.path.join(workdir, "HOG_feature", "*_ObjOut.txt"))):
			fnum=int(re.sub(r'^0+', "", re.sub(r'_.*$', "", os.path.basename(file))))
			with open(file) as f:
				infdata=json.load(f)
			if "bMotionDetected" in infdata and infdata["bMotionDetected"] != 1:
				continue
			if "objs" in infdata:
				for obj in infdata["objs"]:
					if obj["right"] <= obj["left"] or obj["bottom"] <= obj["top"]:
						continue
					outobj={}
					outobj["frame"] = fnum - 1
					outobj["left"] = obj["left"]
					outobj["top"] = obj["top"]
					outobj["width"] = obj["right"] - obj["left"]
					outobj["height"] = obj["bottom"] - obj["top"]
					outobj["motion"] = 1
					outobj["id"] = 1
					ismotion=1
					outfdata["results"]["objects"].append(outobj)
		outfdata["results"]["time"] = datetime.datetime.now().isoformat()
		outfdata["results"]["tags"] = []
		if ismotion == 1:
			outfdata["results"]["tags"].append("motion")
		else:
			outfdata["results"]["tags"].append("still")
		with open(outcfgfile, "w") as f:
			json.dump(outfdata, f, indent=4)

	if "out_dist_yuv" in cfg.keys():
		if cfg["out_dist_yuv"] == 1:
			outyuv_path = os.path.dirname(outfile)
			shutil.copy(infile, outyuv_path)
			output_dist_path = os.path.join(outfile, 'md_data')
			os.makedirs(output_dist_path, exist_ok = True)
			for file in sorted_alphanumeric(glob.glob(os.path.join(workdir, "HOG_feature", "*_Full.csv"))):
				shutil.copy(file, output_dist_path)

	return outcfgfile
