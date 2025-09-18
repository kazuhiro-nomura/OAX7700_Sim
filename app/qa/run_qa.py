#!/usr/bin/python3

import re
import os
import sys
import glob
from pathlib import Path
import shutil
from datetime import date
import datetime
import copy
import json
import platform
import shlex
import csv
import argparse

#################
#   setup Path
#################
pydir=os.path.dirname(os.path.abspath(__file__))

#################
#   setup submodule and config
#################
FF_DIR="md_OA8100"  #FIXME: This FF_DIR doesn't really need a change for different chip.
if platform.system() == "Windows":
	ffmpeg_path=os.path.join(pydir, "..", "..", "lib", f"{FF_DIR}", "bin", "win", "ffmpeg.exe -y")
	ffprobe_path=os.path.join(pydir, "..", "..", "lib", f"{FF_DIR}", "bin", "win", "ffprobe.exe")
else:
	ffmpeg_path=os.path.join(pydir, "..", "..", "lib", f"{FF_DIR}", "bin", "linux", "ffmpeg -y")
	ffprobe_path=os.path.join(pydir, "..", "..", "lib", f"{FF_DIR}", "bin", "linux", "ffprobe")

#################
#   helper functions
#################
# return sequence name of the filename
def get_seqfilename(filename):
	s=os.path.basename(os.path.normpath(filename))
	if os.path.isfile(filename):
		# remove extension
		s=os.path.splitext(s)[0]
	# remove _{W}x{H}
	s=re.sub(r'_\d+x\d+', "", s)
	s=re.sub(r'_P420', "", s)
	s=re.sub(r'_P422', "", s)

	return s

# return sequence name of the filename
def get_resolution(filename):
	w=0
	h=0
	if os.path.isdir(filename):
		cfgfile=os.path.join(filename, "cfg.json")
		if os.path.isfile(cfgfile):
			with open(cfgfile) as f:
				fdata=json.load(f)
			w=int(fdata["width"])
			h=int(fdata["height"])
	elif os.path.isfile(filename):
		if os.path.splitext(filename)[1] == ".yuv":
			# find all _{W}x{H}, use latest one
			for res in re.findall(r'_(\d+)x(\d+)', filename):
				w=int(res[0])
				h=int(res[1])
			# if not find, open the xxx.yuv.json file
			if w==0 or h==0:
				cfgfile=filename + ".json"
				if os.path.isfile(cfgfile):
					with open(cfgfile) as f:
						fdata=json.load(f)
					w=int(fdata["width"])
					h=int(fdata["height"])
		else:
			# use ffprobe to get resolution
			output_dir="."
			resolution_tmpfile=os.path.join(output_dir, "input_resolution.txt")
			cmd="{0} -v error -select_streams v:0 -show_entries stream=width,height -of default=nw=1:nk=1 {1} > {2}".format(ffprobe_path, filename, resolution_tmpfile)
			os.system(cmd)
			f=open(resolution_tmpfile)
			w = int(f.readline())
			h = int(f.readline())
			f.close()
			os.remove(resolution_tmpfile)

	return w,h

def md_get_resolution(md_config_file):
	w=0
	h=0
	f=open(md_config_file)
	while True:
		s = f.readline()
		if not s:
			break
		# remove trailing #
		s=re.sub("#.*$", "", s)
		s=re.findall("\s*(\w+)\s*=\s*(\w+)\s*", s)
		for kv in s:
			if kv[0] == "MD_ScaledW":
				w=int(kv[1])
			if kv[0] == "MD_ScaledH":
				h=int(kv[1])
	f.close()

	return w,h

def module_exec(infile, outfile, workdir, module_cfg):
	module_py_file=module_cfg["module"]

	# convert relative dir to absolute dir
	for fkey in module_cfg["param"]:
		if isinstance(module_cfg["param"][fkey], str) and module_cfg["param"][fkey].startswith("config/"):
			module_cfg["param"][fkey]=os.path.abspath(module_cfg["param"][fkey])
	# import module
	sys.path.insert(0, os.path.dirname(os.path.join(pydir, module_py_file)))
	py_imported = __import__(os.path.splitext(os.path.basename(module_py_file))[0])

	# run module.process()
	return py_imported.module_process(infile, outfile, workdir, module_cfg)

def run_step(sim_cfg, step):
	if "in" in sim_cfg[step]:
		in_step=sim_cfg[step]["in"]
	else:
		in_step="HEAD"
	sim_cfg[step]["in_width"] = sim_cfg[in_step]["out_width"]
	sim_cfg[step]["in_height"] = sim_cfg[in_step]["out_height"]
	cur_infile = sim_cfg[in_step]["real_outfile"]
	if "in1" in sim_cfg[step]:
		# some module need two inputs
		sim_cfg[step]["in1_file"] =   sim_cfg[sim_cfg[step]["in1"]]["real_outfile"]
		sim_cfg[step]["in1_width"] =  sim_cfg[sim_cfg[step]["in1"]]["out_width"]
		sim_cfg[step]["in1_height"] = sim_cfg[sim_cfg[step]["in1"]]["out_height"]
	if "frame" not in sim_cfg[step]:
		if "frame" in sim_cfg[in_step]:
			sim_cfg[step]["frame"] = sim_cfg[in_step]["frame"]
	if "id" not in sim_cfg[step]:
		if "id" in sim_cfg[in_step]:
			sim_cfg[step]["id"] = sim_cfg[in_step]["id"]

	# if output width/height not set, force it same as input.
	# so it means yuvscale must set the out_width and out_height.
	# This generic flow won't get it from next step's input resolution
	if "out_width" not in sim_cfg[step]:
		sim_cfg[step]["out_width"] = sim_cfg[step]["in_width"]
	if "out_height" not in sim_cfg[step]:
		sim_cfg[step]["out_height"] = sim_cfg[step]["in_height"]

	# output to $output_dir/$runname/$seqdirname/$seqfilename, and module add extension and add {W}x{H}, P422, etc
	# workdir to $workdir/$runname/$seqdirname/$seqfilename/
	step_outfile=os.path.join(sim_cfg["output_dir"], sim_cfg[step]["runname"], sim_cfg["seqdirname"])
	if not os.path.isdir(step_outfile):
		os.makedirs(step_outfile, exist_ok = True)
	step_outfile=os.path.join(step_outfile, sim_cfg["seqfilename"])
	step_workdir=os.path.join(sim_cfg["workdir"], sim_cfg[step]["runname"], sim_cfg["seqdirname"], sim_cfg["seqfilename"])
	if not os.path.isdir(step_workdir):
		os.makedirs(step_workdir, exist_ok = True)

	sim_cfg[step]["ffmpeg_path"] = ffmpeg_path

	if "force" in sim_cfg and sim_cfg["force"] == 1:
		sim_cfg[step]["force"] = 1

	sim_cfg[step]["real_outfile"]=module_exec(cur_infile, step_outfile, step_workdir, sim_cfg[step])
	if sim_cfg[step]["real_outfile"] == "":
		print("[ERROR] step {0} failed".format(step))
		exit(1)

	# update json file
	if sim_cfg[step]["real_outfile"].endswith(".json"):
		cfgfile = sim_cfg[step]["real_outfile"]
	else:
		cfgfile = sim_cfg[step]["real_outfile"] + ".json"
	if os.path.isfile(cfgfile):
		with open(cfgfile) as f:
			fdata=json.load(f)
		if "orig_filename" not in fdata:
			if sim_cfg["dataset_dir"] != "":
				fdata["orig_filename"] = os.path.join(sim_cfg["input_runname"], sim_cfg["input_seqfullname"])
			else:
				fdata["orig_filename"] = sim_cfg["input_seqfullname"]
		if in_step != "HEAD":
			# TODO: remove leading output_dir
			fdata["prev_filename"] = cur_infile
		fdata["width"] = sim_cfg[step]["out_width"]
		fdata["height"] = sim_cfg[step]["out_height"]
		fdata["last_update_time"] = datetime.datetime.now().isoformat()
		with open(cfgfile, "w") as f:
			json.dump(fdata, f, indent=4)

	return cfgfile

def run_step_chain(sim_cfg, cur_step):
	# run self
	ret = run_step(sim_cfg, cur_step)
	# run child
	if "out" in sim_cfg[cur_step] and sim_cfg[cur_step]["out"] != "":
		next_step = sim_cfg[cur_step]["out"]
		if cur_step.startswith("remapmd"):
			# for MD, need foreach objects[] and call next steps
			with open(sim_cfg[cur_step]["real_outfile"] + ".json") as f:
				mddata=json.load(f)
			md_result=mddata["results"]
			for obj in md_result["objects"]:
				# use new data for each objects
				sim_cfg_new = copy.deepcopy(sim_cfg)
				# NOTE: the yuvscale support yuvsequence input, and extract frame N to do scale down
				sim_cfg_new[next_step]["frame"] = obj["frame"]
				sim_cfg_new[next_step]["id"] = obj["id"]
				# yuvscale simulator will use these data to do YUVScale
				sim_cfg_new[next_step]["precrop_left"] = obj["left"]
				sim_cfg_new[next_step]["precrop_top"] = obj["top"]
				sim_cfg_new[next_step]["precrop_right"] = obj["left"] + obj["width"]
				sim_cfg_new[next_step]["precrop_bottom"] = obj["top"] + obj["height"]
		
				ret = run_step_chain(sim_cfg_new, next_step)
		else:
			ret = run_step_chain(sim_cfg, next_step)
	# update brothers
	while "next" in sim_cfg[cur_step]:
		ret = run_step_chain(sim_cfg, sim_cfg[cur_step]["next"])
		cur_step = sim_cfg[cur_step]["next"]

	return ret

def run_one_seq(sim_cfg_orig, input_seqfullname, runname="original"):
	# for each sequence, need reset sim_cfg
	sim_cfg=copy.deepcopy(sim_cfg_orig)
	# get input filename
	infile=input_seqfullname
	seqdirname=os.path.basename(os.path.dirname(input_seqfullname))
	seqfilename=get_seqfilename(input_seqfullname)

	sim_cfg["seqdirname"]=seqdirname
	sim_cfg["seqfilename"]=seqfilename
	sim_cfg["input_seqfullname"]=input_seqfullname
	sim_cfg["input_runname"]=runname

	# get input's resolution
	in_W, in_H=get_resolution(infile)
	sim_cfg["HEAD"] = {}
	sim_cfg["HEAD"]["out_width"] = in_W
	sim_cfg["HEAD"]["out_height"] = in_H
	sim_cfg["HEAD"]["real_outfile"] = infile

	return run_step_chain(sim_cfg, sim_cfg["head"])

def sim_cfg_update_chain(sim_cfg, cur, prev_step):
	# update self
	sim_cfg[cur]["in"] = prev_step
	# update child
	if "out" in sim_cfg[cur] and sim_cfg[cur]["out"] != "":
		sim_cfg_update_chain(sim_cfg, sim_cfg[cur]["out"], cur)
	# update brothers
	while "next" in sim_cfg[cur]:
		sim_cfg_update_chain(sim_cfg, sim_cfg[cur]["next"], prev_step)
		cur = sim_cfg[cur]["next"]

def sim_cfg_update(sim_cfg):
	sim_cfg_update_chain(sim_cfg, sim_cfg["head"], "HEAD")

#################
#   real process start from here
#################

def parse_args():
	parser = argparse.ArgumentParser(description= 'QA for Chip Simulator')
	parser.add_argument('-c', '--cfg', default="sim_cfg.json", required=False, help = 'top level sim_cfg.json file')
	parser.add_argument('-p', '--params', required=False, help='parameters overwrite sim_cfg.json, eg: force=1,output_dir=output1')
	return parser.parse_args()

# parse arg
args = parse_args()

# load params
with open(args.cfg) as f:
	sim_cfg_orig=json.load(f)

# update params
if args.params != None:
	for para in args.params.split(','):
		key, value = para.split('=')
		names = key.split('.')
		target = sim_cfg_orig
		for nm in names[:-1]:
			target = target[nm]
		if value.isnumeric() is True:
			value = int(value)
		target[names[-1]] = value

if "workdir" not in sim_cfg_orig:
	sim_cfg_orig["workdir"]=os.path.join(sim_cfg_orig["output_dir"], "_workdir")

sim_cfg_orig["output_dir"] = os.path.abspath(sim_cfg_orig["output_dir"])
sim_cfg_orig["workdir"] = os.path.abspath(sim_cfg_orig["workdir"])

sim_cfg_update(sim_cfg_orig)

# create output dir
if not os.path.isdir(sim_cfg_orig["output_dir"]):
	os.makedirs(sim_cfg_orig["output_dir"])

# load sequence list
seq_list = []
if sim_cfg_orig["filelist"].endswith(".csv"):
	sl_csv = os.path.join(os.path.dirname(args.cfg), sim_cfg_orig["filelist"])
	with open(sl_csv) as f:
		sl_data = csv.reader(f)
		seq_list = [seq[0] for seq in sl_data]
else:
	if os.path.isabs(sim_cfg_orig["filelist"]) == True:
		seq_list.append(sim_cfg_orig["filelist"])
	elif os.path.exists(os.path.normpath(os.path.join(sim_cfg_orig["dataset_dir"], sim_cfg_orig["filelist"]))):
		seq_list.append(os.path.normpath(os.path.join(sim_cfg_orig["dataset_dir"], sim_cfg_orig["filelist"])))
	else:
		seq_list.append(os.path.normpath(os.path.join(os.getcwd(), sim_cfg_orig["filelist"])))

# process
result_list = []
for seq in seq_list:
	if os.path.isabs(seq) == True:
		seq_absdir=seq
	elif os.path.isabs(sim_cfg_orig["dataset_dir"]) == True:
		seq_absdir=os.path.join(sim_cfg_orig["dataset_dir"], seq)
	else:
		seq_absdir=os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(args.cfg)), sim_cfg_orig["dataset_dir"], seq))

	if os.path.exists(seq_absdir) == False:
		print("[ERROR] {0} not exist".format(seq_absdir))
		exit(1)

	resultfile = run_one_seq(sim_cfg_orig, seq_absdir)
	result_list.append(resultfile)

# final result list QA
if len(result_list) > 1 and "list_qa" in sim_cfg_orig:
	module_exec(result_list, sim_cfg_orig["output_dir"], "", sim_cfg_orig["list_qa"])
