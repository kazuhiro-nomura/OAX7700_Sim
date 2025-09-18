import os
import re
import sys
import shutil
import json

def module_process(infile, outfile, workdir="", cfg={}):
	outdir = os.path.dirname(outfile)
	outfilename = os.path.splitext(os.path.basename(infile))[0]
	outfilename = re.sub(r'_P420', "", outfilename)
	outfilename = re.sub(r'_P422', "", outfilename)
	outfilename = re.sub(r'_P400', "", outfilename)
	outfile = os.path.join(outdir, outfilename + ".rgb")

	if "force" in cfg and cfg["force"] == 0:
		if os.path.isfile(outfile):
			print("[INFO] skip YUV2RGB because output file exist")
			return outfile

	# TODO: check infile name to get input format, _P420 or _P422

	# match HW YUV2RGB
	# r = int((0x25 * (y - 0x10) + 0x0  * (u - 0x80) + 0x33 * (v - 0x80) + 0x1f)) >> 5
	# g = int((0x25 * (y - 0x10) - 0xd  * (u - 0x80) - 0x1a * (v - 0x80) + 0x1f)) >> 5
	# b = int((0x25 * (y - 0x10) + 0x41 * (u - 0x80) + 0x0  * (v - 0x80) + 0x1f)) >> 5
	# and then limit to (0, 255)
	w = cfg["in_width"]
	h = cfg["in_height"]
	with open(infile, "rb") as f:
		yuv = bytearray(f.read())

	yuv_frm_size = w * h * 2
	rgb_frm_size = w * h * 3
	assert len(yuv) % yuv_frm_size == 0, "YUV file corrupted, size {} is not a multiple of frame size {} (width {} height {})".format(
		len(yuv), yuv_frm_size, w, h)

	num_frame = int(len(yuv) / yuv_frm_size)
	rgb = bytearray(rgb_frm_size * num_frame)

	for ifrm in range(num_frame):
		for ih in range(0, h):
			for iw in range(0, w):
				y = yuv[ifrm * yuv_frm_size + ih * w + iw]
				u = yuv[ifrm * yuv_frm_size + int(h * w + ih * w / 2 + iw / 2)]
				v = yuv[ifrm * yuv_frm_size + int(h * w * 3 / 2 + ih * w / 2 + iw / 2)]
				r = int((0x25 * (y - 0x10) + 0x0  * (u - 0x80) + 0x33 * (v - 0x80) + 0x1f)) >> 5
				g = int((0x25 * (y - 0x10) - 0xd  * (u - 0x80) - 0x1a * (v - 0x80) + 0x1f)) >> 5
				b = int((0x25 * (y - 0x10) + 0x41 * (u - 0x80) + 0x0  * (v - 0x80) + 0x1f)) >> 5
				if r < 0:
					r = 0
				if r > 255:
					r = 255
				if g < 0:
					g = 0
				if g > 255:
					g = 255
				if b < 0:
					b = 0
				if b > 255:
					b = 255
				rgb[ifrm * rgb_frm_size + ih * w * 3 + iw * 3] = r
				rgb[ifrm * rgb_frm_size + ih * w * 3 + iw * 3 + 1] = g
				rgb[ifrm * rgb_frm_size + ih * w * 3 + iw * 3 + 2] = b
	with open(outfile, "wb") as f:
		f.write(rgb)

	incfgfile = infile + ".json"
	outcfgfile = outfile + ".json"
	if os.path.isfile(incfgfile):
		shutil.copyfile(incfgfile, outcfgfile)

	# ffmpeg to convert
	# cmd="{0} -f rawvideo -pix_fmt yuv422p -s:v {1}x{2} -i {3} -c:v rawvideo -pix_fmt rgb24 {4}_ffmpeg.rgb".format(cfg["ffmpeg_path"], cfg["in_width"], cfg["in_height"], infile, outfile)
	# print(cmd)
	# os.system(cmd)

	return outfile
