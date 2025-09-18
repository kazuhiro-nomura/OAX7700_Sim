# OA7700 ISP simulator

## usage


`bin/isp_sim_OA7700.py -i bcap20200723_114918_65.raw -o 2/output/bcap20200723_114918_65.yuv -dbgpath 2/output/ -lp 2/load_para.txt -w 640 -h 360 -b 8 -gain 0x10 -blc 0x10 -en 1 -dbc 0 -binc 1 -nFlip 1 -nMirror 0 -nControl 3 -nIfNewFilter 0 -gamma 2 -gamma_lut 2/gamma.txt -m_nAvgXstart 16 -m_nAvgYstart 16 -m_nAvgXwin 608 -m_nAvgYwin 320 -AvgWtFile 2/avg_wt.txt -awb 0 -m_nAWBGainB 0xa0 -m_nAWBGainGB 0x80 -m_nAWBGainGR 0x80 -m_nAWBGainR 0x170 -m_nAvgMinVal 4 -m_nAvgMaxVal 247 -Y_b_coeff 29 -Y_g_coeff 150 -Y_r_coeff 77 -U_b_coeff 128 -U_g_coeff -85 -U_r_coeff -43 -V_b_coeff -21 -V_g_coeff -107 -V_r_coeff 128 -U_BIAS 0 -V_BIAS 0`

### parameters

| param | type | default | desc |
| ----- | ---- | ------- | ---- |
| i | filename | input_file.raw | input file name, RAW |
| o | filename | output_file.yuv | output file name, yuv |
| dbgpath | dirname | debug_dir | output intermediate directory |
| lp | filename | load_para.txt | parameter txt file |
| w | integer | 0 | width |
| h | integer | 0 | height |
| gain | integer | 0x10 | gain |
| blc | integer | 0 | blc |
| awb | 0,1 | 1 | enable Auto White Balance, from R/G/B statistics using grayworld algo |
| m_nAWBGainB | integer | 0x80 | manual white balance, used when awb==0 |
| m_nAWBGainGB | integer | 0x80 | manual white balance, used when awb==0 |
| m_nAWBGainGR | integer | 0x80 | manual white balance, used when awb==0 |
| m_nAWBGainR | integer | 0x80 | manual white balance, used when awb==0 |
| dpc | 0,1 | 1 | Dead Pixel Compensation Enable |
| binc | 0,1 | 1 | Binning Correction Enable |
| nControl | 1,2,3,4 | 3 | Binning Correction nControl |
| nIfNewFilter | 0,1 | 0 | Binning Correction nIfNewFilter |
| nFlip | 0,1 | 1 | Binning Correction nFlip |
| nMirror | 0,1 | 0 | Binning Correction nMirror |
| gamma | 0,2 | 2 | 0: disable, 2: RGB Gamma |
| gamma_lut | filename | gamma_lut.txt | gamma LUT file name |
| m_nAvgXstart | integer | 0 | AECAGC X start |
| m_nAvgYstart | integer | 0 | AECAGC Y start |
| m_nAvgXwin | integer | 640 | AECAGC X window size |
| m_nAvgYwin | integer | 360 | AECAGC Y window size |
| m_nAvgMinVal | integer | 4 | AECAGC min pixel Val |
| m_nAvgMaxVal | integer | 247 | AECAGC max pixel Val |
| AvgWtFile | filename | avg_wt.txt | AECAGC weight file |
| Y_b_coeff | integer | 29  | Y_b_coeff |
| Y_g_coeff | integer | 150 | Y_g_coeff |
| Y_r_coeff | integer | 77  | Y_r_coeff |
| U_b_coeff | integer | 128 | U_b_coeff |
| U_g_coeff | integer | -85 | U_g_coeff |
| U_r_coeff | integer | -43 | U_r_coeff |
| V_b_coeff | integer | -21 | V_b_coeff |
| V_g_coeff | integer | -107 | V_g_coeff |
| V_r_coeff | integer | 128 | V_r_coeff |
| UV_BIAS | integer | 0 | UV_BIAS |
| U_BIAS | integer | 0 | U_BIAS |
| V_BIAS | integer | 0 | V_BIAS |

The parameters can be in command line, or in json file. The priorities are:

1. command line arguments has highest priority, run `bin/isp_sim_OA7700.py` to see what parameters supported.
2. if some arguments not specified in command line argument, then application will look for xxx.json file exist (xxx is the file name of input raw file without extension) in the same directory as raw file, the arguments in this file will be used.
3. if there's some arguments not specified in previous two stpes, the "cfg.json" file in the same directory as raw file will be used. The benefits for cfg.json is this file can be applied to all the raw files in the same directory.
4. at last, if still some arguments not set, the default value in `bin/isp_sim_OA7700.py` file will be used. look at the "param_default" in `bin/isp_sim_OA7700.py`

### json file example

this is one example of json file(xxx.json, or cfg.json):


```
{
	"w": "640",
	"h": "360",
	"b": "8",
	"nFlip": "1",
	"gain": "0x10",
	"blc": "0x10"
}
```
