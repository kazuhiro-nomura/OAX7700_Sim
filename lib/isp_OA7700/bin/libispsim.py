#!/usr/bin/python3

import re
import os
import sys
import glob
import copy
from pathlib import Path
import shutil
from datetime import date
from array import array
import struct
import json
import platform
import argparse

def sorted_alphanumeric(data):
	convert = lambda text: int(text) if text.isdigit() else text.lower()
	alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
	return sorted(data, key=alphanum_key)

def isp_process_one_file(isp_exe, param_cur, oneinputfile, oneoutputfile):
	if oneinputfile.endswith(".vrf"):
		# convert VRF to RAW
		rawoutputfile = os.path.splitext(oneoutputfile)[0] + ".raw"
		try:
			fp = open(oneinputfile, "rb")
		except IOError:
			print(f"open file {oneinputfile} fail")
			exit(1)
		fp.seek(-128, 2)
		vrfmeta = array('I')
		vrfmeta.fromfile(fp, 32)
		fp.close()

		vrf_width = vrfmeta[0] & 0xffff
		vrf_height = vrfmeta[0] >> 16
		bpp_flag = (vrfmeta[30] & 0xff00) >> 8
		if bpp_flag == 0:
			vrf_bpp = 10
		elif bpp_flag == 1:
			vrf_bpp = 12
		elif bpp_flag == 2:
			vrf_bpp = 14
		elif bpp_flag == 3:
			vrf_bpp = 16
		else:  # vrf_bpp ==4
			vrf_bpp = 8
		if vrf_bpp > 8:
			fp = open(oneinputfile, "rb")
			rawData = struct.unpack(f"{vrf_width*vrf_height}H", fp.read(2*vrf_width*vrf_height))
			fp.close()
			if vrf_bpp == 12:
				rawData = [ x >> 4 for x in rawData ]
			elif vrf_bpp == 10:
				rawData = [ x >> 2 for x in rawData ]
			elif vrf_bpp == 14:
				rawData = [ x >> 6 for x in rawData ]
		else:
			fp = open(oneinputfile, "rb")
			rawData = struct.unpack(f"{vrf_width*vrf_height}B", fp.read(vrf_width*vrf_height))
			fp.close()
		fp = open(rawoutputfile, "wb")
		rawd = struct.pack(f"{vrf_width*vrf_height}B", *rawData)
		fp.write(rawd)
		fp.close()
		oneinputfile = rawoutputfile
	# generate command args
	cmdargs = ""
	for fkey in param_cur:
		if fkey != "--":
			cmdargs += " -{0} {1}".format(fkey, param_cur[fkey])
		else:
			cmdargs += " {0}".format(param_cur[fkey])

	cmd="{0} -i {1} -o {2} {3}".format(isp_exe, oneinputfile, oneoutputfile, cmdargs)
	print("[INFO] Processing {0}".format(cmd.replace("\\", "\\\\")))

	os.system(cmd)

def isp_process(chip, infile, outfile, param_cur):
	# overwrite by cfg.json in the same directory
	if os.path.isdir(infile):
		cfgfile=os.path.join(infile, "cfg.json")
	else:
		# look for {infile}.json first
		cfgfile=infile + ".json"
		if not os.path.isfile(cfgfile):
			# if not exist, look for cfg.json in the same directory
			cfgfile=os.path.join(os.path.dirname(infile), "cfg.json")
	if os.path.isfile(cfgfile):
		f=open(cfgfile)
		fdata=json.load(f)
		f.close()
		if "width" in fdata:
			param_cur["w"] = fdata["width"]
		if "height" in fdata:
			param_cur["h"] = fdata["height"]
		if "isp" in fdata:
			for fkey in fdata["isp"]:
				param_cur[fkey] = fdata["isp"][fkey]

	pydir=os.path.dirname(os.path.abspath(__file__))
	if platform.system() == "Windows":
		isp_exe=os.path.join(pydir, "win", "isp_sim_{0}.exe".format(chip))
	else:
		isp_exe=os.path.join(pydir, "linux", "isp_sim_{0}".format(chip))

	if os.path.isfile(infile):
		# process file
		isp_process_one_file(isp_exe, param_cur, infile, outfile)
	else:
		filelist=sorted_alphanumeric(glob.glob(os.path.join(infile, "*.raw")))
		if len(filelist) == 0:
			filelist=sorted_alphanumeric(glob.glob(os.path.join(infile, "*.vrf")))
			if len(filelist) == 0:
				print(f'ERROR: {infile}/*.raw and {infile}/*.vrf not exist')
				exit(-1)

		if os.path.isfile(outfile):
			os.remove(outfile)
		outf=open(outfile, 'wb')

		for f in filelist:
			tmpfile=os.path.join(param_cur["dbgpath"], os.path.splitext(os.path.basename(f))[0] + ".yuv")
			isp_process_one_file(isp_exe, param_cur, f, tmpfile)
			yuvf=open(tmpfile, 'rb')
			outf.write(yuvf.read())
			yuvf.close()

		outf.close()

def libisp_process_fromargs(chip, param_default):
	parser = argparse.ArgumentParser(description='ISP Simulator for {0}'.format(chip), add_help=False)
	parser.add_argument('-i', metavar='inputfile', required=True, help='input RAW filename')
	parser.add_argument('-o', metavar='outputfile', required=True, help='output YUV filename')
	for fkey in param_default:
		if param_default[fkey][0] != "":
			parser.add_argument("-{0}".format(fkey), metavar=param_default[fkey][1], help=param_default[fkey][2])
	parser.add_argument('rest', nargs=argparse.REMAINDER)

	if len(sys.argv) == 1:
		parser.print_help()
	args = parser.parse_args()

	inputfile=getattr(args, "i")
	outputfile=getattr(args, "o")

	# if inputdir is a directory, then process all the *.raw file in this
	# if inputdir is a file, then only process this file

	param_cur = {}
	# load default value
	for fkey in param_default:
		param_cur[fkey] = param_default[fkey][0]
	# set param from the command args
	for fkey in param_default:
		if param_default[fkey][0] != "" and getattr(args, fkey) != None:
			param_cur[fkey] = getattr(args, fkey)
	# add rest command
	param_rest = getattr(args, "rest")
	if param_rest != None:
		param_cur["--"] = ""
		for v in param_rest:
			if v != "--":
				param_cur["--"] += " {0}".format(v)

	isp_process(chip, inputfile, outputfile, param_cur)

# API for module call
def libisp_module_process(chip, param_default, infile, outfile, workdir="", cfg={}):
	# if inputdir is a directory, then process all the *.raw file in this
	# if inputdir is a file, then only process this file

	param=cfg["param"]

	param_cur = {}
	# load default value
	for fkey in param_default:
		param_cur[fkey] = param_default[fkey][0]

	# overwrite by the param{} from this API
	for fkey in param:
		param_cur[fkey] = param[fkey]
	if workdir != "":
		param_cur["dbgpath"] = workdir

	outfile += "_{0}x{1}_P422.yuv".format(cfg["in_width"], cfg["in_height"])

	if "force" in cfg and cfg["force"] == 0:
		if os.path.isfile(outfile):
			print("[INFO] skip ISP because output file exist")
			return outfile

	if os.path.isdir(infile):
		isp_process(chip, infile, outfile, param_cur)
		cfgfile=os.path.join(infile, "cfg.json")
		if os.path.isfile(cfgfile):
			outcfgfile=outfile + ".json"
			shutil.copyfile(cfgfile, outcfgfile)
			outcfgfile=os.path.join(param_cur["dbgpath"], "cfg.json")
			shutil.copyfile(cfgfile, outcfgfile)
	else:
		# it's file
		if os.path.splitext(infile)[1] == ".yuv":
			# yuv file, just copy
			shutil.copyfile(infile, outfile)
		else:
			# ffmpeg to convert
			cmd="{0} -i {1} -c:v rawvideo -pix_fmt yuv422p {2}".format(cfg["ffmpeg_path"], infile, outfile)
			os.system(cmd)

		# update json file, just copy
		incfgfile=infile + ".json"
		if os.path.isfile(incfgfile):
			outcfgfile=outfile + ".json"
			shutil.copyfile(incfgfile, outcfgfile)

	return outfile
