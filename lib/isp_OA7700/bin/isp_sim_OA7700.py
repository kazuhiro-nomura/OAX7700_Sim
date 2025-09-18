#!/usr/bin/python3

import os
import sys
import platform
from libispsim import *

param_default_OA7700={
  "w": ["640", "width", "input raw image width"],
  "h": ["360", "height", "input raw image height"],
  "b": ["8", "bit", "input raw image bits"],
  "gain": ["0x10", "gain", "raw image analog gain"],
  "blc": ["0x10", "blc", "raw image BLC"],
  "en": ["1", "", "enable processing"],
  "nFlip": ["1", "flip", "1 to enable flip"],
  "nMirror": ["0", "mirror", "1 to enable mirror"],
  "nControl": ["3", "control", ""],
  "nIfNewFilter": ["0", "filter", ""],
  "lp": ["load_para.txt", "para_file", "para file"],
  "gamma": ["2", "", "enable gamma"],
  "gamma_lut": ["gamma.txt", "gamma_file", "gamma lut file"],
  "dbgpath": ["output", "debug_path", "debug output path"],
  "binc": ["1", "", "enable BinningCorrection"],
  "dpc": ["1", "", "enable Dead Pixel Correction"],
  "awb": ["1", "", "enable AWB"],
  "m_nAWBGainB": ["0x80", "WBGainB", "WBGainB, 0x80 based"],
  "m_nAWBGainGB": ["0x80", "WBGainB", "WBGainB, 0x80 based"],
  "m_nAWBGainGR": ["0x80", "WBGainB", "WBGainB, 0x80 based"],
  "m_nAWBGainR": ["0x80", "WBGainB", "WBGainB, 0x80 based"],
  "m_nAvgXstart": ["0", "AvgXstart", "AECAGC X start"],
  "m_nAvgYstart": ["0", "AvgYstart", "AECAGC Y start"],
  "m_nAvgXwin": ["640", "AvgXwin", "AECAGC X window size"],
  "m_nAvgYwin": ["360", "AvgXwin", "AECAGC Y window size"],
  "m_nAvgMinVal": ["4", "AvgMinVal", "AECAGC min pixel Val"],
  "m_nAvgMaxVal": ["247", "AvgMaxVal", "AECAGC max pixel Val"],
  "AvgWtFile": ["avg_wt.txt", "AvgWtFile", "AECAGC weight file"],
  "Y_b_coeff": ["29",  "Y_b_coeff", "Y_b_coeff"],
  "Y_g_coeff": ["150", "Y_g_coeff", "Y_g_coeff"],
  "Y_r_coeff": ["77",  "Y_r_coeff", "Y_r_coeff"],
  "U_b_coeff": ["128", "U_b_coeff", "U_b_coeff"],
  "U_g_coeff": ["-85", "U_g_coeff", "U_g_coeff"],
  "U_r_coeff": ["-43", "U_r_coeff", "U_r_coeff"],
  "V_b_coeff": ["-21", "V_b_coeff", "V_b_coeff"],
  "V_g_coeff": ["-107", "V_g_coeff", "V_g_coeff"],
  "V_r_coeff": ["128", "V_r_coeff", "V_r_coeff"],
  "UV_BIAS": ["0", "UV_BIAS", "UV_BIAS"],
  "U_BIAS": ["0", "U_BIAS", "U_BIAS"],
  "V_BIAS": ["0", "V_BIAS", "V_BIAS"],
}

def isp_process_fromargs():
	libisp_process_fromargs("OA7700", param_default_OA7700)

def module_process(infile, outfile, workdir="", cfg={}):
	return libisp_module_process("OA7700", param_default_OA7700, infile, outfile, workdir, cfg)

if __name__ == '__main__':
	isp_process_fromargs()
