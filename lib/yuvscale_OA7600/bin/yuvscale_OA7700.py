#!/usr/bin/python3

import os
import re
import shutil
import sys
import platform
import copy
import json

# return HScale, VScale
def yuvscale_calc(in_w, in_h, out_w, out_h):
	if out_w > in_w:
		print("[ERROR] YUVScale not support scale up W {0}->{1}".format(in_w, out_w))
		exit(1)
	if out_h > in_h:
		print("[ERROR] YUVScale not support scale up H {0}->{1}".format(in_h, out_h))
		exit(1)
	w_ratio = int((out_w * 128 + (in_w - 1))/ in_w)
	out_temp_w = w_ratio * in_w / 128;
	if out_temp_w < out_w:
		print("[ERROR] yuvscale W ratio calc error: {0}, {1}, {2}, {3}".format(in_w, out_w, w_ratio, out_temp_w))
		exit(1)
	h_ratio = int((out_h * 128 + (in_h - 1))/ in_h)
	out_temp_h = h_ratio * in_h / 128;
	if out_temp_h < out_h:
		print("[ERROR] yuvscale H ratio calc error: {0}, {1}, {2}, {3}".format(in_h, out_h, h_ratio, out_temp_h))
		exit(1)
	return w_ratio, h_ratio

def yuvscale_process(infile, outfile, in_w, in_h, dbgpath, myargs):

	pydir=os.path.dirname(os.path.abspath(__file__))
	if platform.system() == "Windows":
		yuvscale_exe=os.path.join(pydir, "win", "yuvscale_sim_OA7700.exe")
	else:
		yuvscale_exe=os.path.join(pydir, "linux", "yuvscale_sim_OA7700")

	fsize=os.path.getsize(infile)
	if fsize < in_w*in_h*2*2:
		# < two YUV file, single file
		for i in myargs:
			yuvscale_exe += " {0}".format(i.replace(' ', '\\ '))
		os.system(yuvscale_exe)
		return

	# sequence file
	# input: dbgpath/in_FNn.yuv
	# output: dbgpath/out_FNn.yuv
	# dbgpath: dbgpath/FNn/
	# combine output: outfile
	f=open(infile, mode="rb")
	fwout=open(outfile, mode="wb")
	cnt=0
	while True:
		b=f.read(in_w*in_h*2)
		if not b:
			break
		cur_infile=os.path.join(dbgpath, "in_FN{0}.yuv".format(cnt))
		cur_outfile=os.path.join(dbgpath, "out_FN{0}.yuv".format(cnt))
		cur_dbgpath=os.path.join(dbgpath, "FN{0}".format(cnt))
		os.makedirs(cur_dbgpath, exist_ok = True)
		fw=open(cur_infile, mode="wb")
		fw.write(b)
		fw.close()
		# exec one file
		cur_args=copy.deepcopy(myargs)
		for i in range(0, len(cur_args)):
			if cur_args[i] == "-i":
				cur_args[i+1]=cur_infile
			if cur_args[i] == "-o":
				cur_args[i+1]=cur_outfile
			if cur_args[i] == "-dbgpath":
				cur_args[i+1]=cur_dbgpath
		cmd=yuvscale_exe
		for i in cur_args:
			cmd += " {0}".format(i.replace(' ', '\\ '))
		print("exec: {0}".format(cmd.replace("\\", "\\\\")))
		os.system(cmd)
		# append to output file
		fr=open(cur_outfile, mode="rb")
		frb=fr.read()
		fwout.write(frb)
		fr.close()
		cnt+=1
	f.close()

	print("[INFO] total {0} frames processed".format(cnt))

def copy_json_file(infile, outfile):
	incfgfile=infile + ".json"
	if os.path.isfile(incfgfile):
		outcfgfile=outfile + ".json"
		shutil.copyfile(incfgfile, outcfgfile)

def update_json_file(infile, outfile, HScale, VScale):
	incfgfile=infile + ".json"
	if os.path.isfile(incfgfile):
		outcfgfile=outfile + ".json"
		# TODO: scale the distzone
		# NOTE: not scale objects because remapmd will remap ROI to original resolution
		with open(incfgfile) as f:
			fdata=json.load(f)
		if "distzone" in fdata:
			newdz={}
			for k in fdata["distzone"]:
				newdz[k] = [ [int(x[0] * HScale / 128), int(x[1] * VScale / 128)] for x in fdata["distzone"][k] ]
			fdata["distzone"] = newdz
		with open(outcfgfile, "w") as f:
			json.dump(fdata, f, indent=4)

def module_process(infile, outfile, workdir="", cfg={}):
	param=cfg["param"]
	out_width=cfg["out_width"]
	out_height=cfg["out_height"]
	in_width=cfg["in_width"]
	in_height=cfg["in_height"]

	outfile += "_{0}x{1}_P422.yuv".format(cfg["out_width"], cfg["out_height"])

	if "force" in cfg and cfg["force"] == 0:
		if os.path.isfile(outfile):
			print("[INFO] skip yuvscale because output file exist")
			return outfile

	if out_width == in_width and out_height == in_height:
		# just copy
		shutil.copyfile(infile, outfile)
		# update json file, just copy
		copy_json_file(infile, outfile)
		return outfile

	myargs=[]
	myargs.append("-i")
	myargs.append(infile)
	myargs.append("-o")
	myargs.append(outfile)
	myargs.append("-size")
	myargs.append("{0}".format(in_width))
	myargs.append("{0}".format(in_height))
	myargs.append("-dbgpath")
	myargs.append(workdir)

	HScale, VScale = yuvscale_calc(in_width, in_height, out_width, out_height)
	print("[INFO] YUVScale {0}x{1} -> {2}x{3}, ratio {4}, {5}".format(in_width, in_height, out_width, out_height, HScale, VScale))

	# generate new param file
	cfgfilename=os.path.join(workdir, "default_param.txt")
	f=open(param["lp"])
	fw=open(cfgfilename, 'w')
	while True:
		s = f.readline()
		if not s:
			break
		s=re.sub("SED_FOR_HSCALE", "{0}".format(HScale), s)
		s=re.sub("SED_FOR_VSCALE", "{0}".format(VScale), s)
		s=re.sub("SED_FOR_POSTCROP_W", "{0}".format(out_width), s)
		s=re.sub("SED_FOR_POSTCROP_H", "{0}".format(out_height), s)
		fw.write(s)
	f.close()
	fw.close()
	param["lp"]=cfgfilename

	for fkey in param:
		myargs.append("-{0}".format(fkey))
		myargs.append("{0}".format(param[fkey]))

	yuvscale_process(infile, outfile, in_width, in_height, workdir, myargs)

	# update json file, just copy
	update_json_file(infile, outfile, HScale, VScale)

	return outfile

def yuvscale_process_fromargs():
	myargs=copy.deepcopy(sys.argv[1:])

	w=0
	h=0
	infile=None
	outfile=None
	dbgpath="."
	for i in range(0, len(myargs)):
		if myargs[i] == "-size":
			w=int(myargs[i+1])
			h=int(myargs[i+2])
		if myargs[i] == "-i":
			infile=myargs[i+1]
		if myargs[i] == "-o":
			outfile=myargs[i+1]
		if myargs[i] == "-dbgpath":
			dbgpath=myargs[i+1]
	if w==0 or h==0:
		print("[ERROR] -size W H not specified")
		exit(1)

	yuvscale_process(infile, outfile, w, h, dbgpath, myargs)

if __name__ == '__main__':
	yuvscale_process_fromargs()
