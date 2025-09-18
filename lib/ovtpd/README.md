# OVTPD Simulator

## Prerequisite

Please make sure `Quartz` environment is installed and activated.

## Inference Interface

#### Usage

Usage of the `pyinfer.py` interface is as follows:

```bash
usage: pyinfer.py [-h] [--model {160x288.c0,176x304.c0,176x304.c1,256x320.c1,208x256.c1,208x256.c1q2}] [--input_list INPUT_LIST] [--output_dir OUTPUT_DIR] [--conf_thresh CONF_THRESH] [--nms_thresh NMS_THRESH]

Run inference with OVMWorker based on quartz.

optional arguments:
  -h, --help            show this help message and exit
  --model {160x288.c0,176x304.c0,176x304.c1,256x320.c1,208x256.c1,208x256.c1q2}
                        Model specifier, e.g. '176x304.c0'
  --input_list INPUT_LIST
                        Path of input image list file
  --output_dir OUTPUT_DIR
                        Path of output directory
  --conf_thresh CONF_THRESH
                        Thresh of confidence
  --nms_thresh NMS_THRESH
                        Thresh of IOU for NMS
```

#### Example

```bash
cd tests
python ../bin/pyinfer.py --model 160x288.c0 --input_list filelist/images_160x288.example.txt
```

## Module Interface

#### Usage
To call the `ovtpd.py` module from applications, add the "ovtpd" module and pass the parameters as in the following example.

Its input step should generate a RGB array file, such as "yuv2rgb" module.

The input resolution should match with the model.

#### Example

```json
  "ovtpd" : {
    "module" : "../../lib/ovtpd/bin/ovtpd.py",
    "runname" : "ovtpd",
    "force" : 1,
    "param" : {
      "model" : "176x304.c1",
      "conf_thresh" : 0.38,
      "nms_thresh" : 0.4
    }
  }
```
