#!/usr/bin/python3

import os
import sys
import platform
from libmdsim import *

def md_process_fromargs():
	libmd_process_fromargs(sys.argv, "OA8100")

def module_process(infile, outfile, workdir="", cfg={}):
	return libmd_module_process("OA8100", infile, outfile, workdir, cfg)

if __name__ == '__main__':
	md_process_fromargs()
