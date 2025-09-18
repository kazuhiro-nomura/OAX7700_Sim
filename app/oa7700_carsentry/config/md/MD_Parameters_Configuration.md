# CarSentry MD Parameters Configuration

The following table describes the configuration of the CarSentry MD parameters, including configurable parameter, suggested value, range, and remark.

## Main Parameters

| Parameter | Suggested<br>Value | Range | Remark |
|------------|---|---|---------------|
| InputFileType | 1 | 0/1/3 | Input file type |
| InputFilePath | - | - | Input file path<br>  • Filename requirements:<br> - End with `_P420` (YUV420) or `_P422` (YUV422)<br>- InputFileType=1, filename end with: `WxH_P420` or `WxH_P422`<br> • Example: `"../../data/xxx_480x384_P420.yuv"` |
| InputWidth | 480 | - | Input video/image width(W), ignore if InputFileType=1 |
| InputHeight | 384 | - | Input video/image height(H), ignore if InputFileType=1 |
| StartFrameIdx | 0 | 0~N-1 | Start frame index<br> ≤the total frames(N) of the video |
| FramesToBeProcessed | -1 | -1 or <br> 1~N-1 | Number of frames to be processed<br> `-1` =all frames, or else<br>no more than (N-StartFrameIdx) |
| OutputFilePath | "output" | - | Output file path |
| MD_RefFrmIdx | 1 | 1-7 | Frame rate dependent:<br>• ≤10FPS:1<br> • ≤15FPS:3<br> • ≤24FPS:6<br> • ≤30FPS:7<br> changing on the actual situation |
| MD_ScaledW | 480 | ≤640 | 8-pixel aligned |
| MD_ScaledH | 384 | ≤480 | 8-pixel aligned |
| MD_ShiftHistMag2 | 3 | 0-7 | *Not mod* |
| MD_ShiftMetric2_Th | 6 | 0-15 | *Not mod* |
| MD_NeighborCorrelation | 1 | 0/1 | 0: Disable<br>1: Enable, FP reduction<br>*Not mod* |
| MD_ActBlk_MinMaxRatio_Scale | 16 | 1-255 | *Not mod* |
| MD_ActBlkRepeatFrmCntTh | 3 | 0-15 | *Not mod* |
| MD_ActNeighborBlkCntTh | 2 | 0-9 | *Not mod* |
| MD_HogDist_Th_<br>nnAdj_Scale | 4 | 0-31 | *Not mod* |
| MD_HogDist_Th_<br>highActiveAdj_scale | 12 | 0-31 | *Not mod* |
| MD_HogDist_Th_<br>AvgYDiffAdj_Scale | 42 | 0-255 | *Not mod* |
| MD_LowDelayEn | 1 | 0/1 | 0: Disable<br>1: Enable<br>*Not mod* |
| MD_HogDist_Th_<br>lowDelay_shift | 1 | 0-3 | *Not mod* |
| MD_HogDist_Th_<br>lowDelay_m_count | 1 | 0-15 | *Not mod* |
| MD_HogDist_Th_<br>lowDelay_prev_nn_count | 1 | 0-7 | *Not mod* |
| MD_HogDistAvailCheck | 1 | 0/1 | Block effectiveness<br>0: skip, all avaliable<br>1: conditional check<br>*Not mod* |
| MD_HogCellSize | 16 | 8/16 | HOG Cell size |
| MD_HogGradX | "-1,-2,0,<br>2,1" | - | 1×5 GradX kernel, 1x5 only<br>*Not mod* |
| MD_HogGradY | "-1,-2,0,<br>2,1" | - | 1×5 GradX kernel, 1x5 only<br>*Not mod* |
| MD_HogMagnitudeWeightEn | 1 | 0/1 | Weighted gradient magnitude<br>0: Disable<br>1: Enabl.<br>*Not mod* |
| MD_HogSmoothBinsEn | 1 | 0/1 | Smoothing filter to the histogram bins<br>0: Disable<br>1: Enable<br>*Not mod* |
| MD_HogNormalizeMode | 0 | 0/1/2 | HOG feature normalization<br>0: Disable<br>1: Enable, output hist norm<br>2: Enable, output sum of gradient magnitude<br>*Not mod* |
| MD_HogDist_<br>AvgYDiffThre_Scale | 8 | 0-255 | Higher=↓FP↑FN |
| MD_HogDist_WeighMethod | 0 | 0/1 | Weighting method for distance<br>0: curBlkVar +<br> abs(curBlkVar - refBlkVar)<br>1: curBlkVar + MAX(curBlkVar-refBlkVar, 0) |
| MD_HogDist_Scale2 | 15 | 0-255 | Higher=↓FP↑FN |
| MD_HogDist_Th | 20 | 0-255 | Higher=↓FP↑FN  |

**Legend:**  
FP=False Positive, FN=False Negative  
*Not mod*=Not recommended to modify
