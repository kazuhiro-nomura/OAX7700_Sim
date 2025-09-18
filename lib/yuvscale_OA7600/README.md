# YUVScale Simulator

## usage

`bin/yuvscale_OA7700.py -i input.yuv -o output.yuv -size W H -b 8 -lp param.txt -dbgpath output/`

input.yuv: input YUV422P format file
output.yuv: output YUV422P format file
W: input image width
H: input image height
param.txt: param for YUVScale, see tests/1/default_param.txt for reference
output: specify the intermediate result output path.
   Note the output.yuv will not be in this output/ directory, so output.yuv must specify the directory if want.

### example

bin/yuvscale_OA7700.py -i input.yuv -o output.yuv -size 1920 1080 -b 8 -lp default_param.txt -dbgpath 1/output/
