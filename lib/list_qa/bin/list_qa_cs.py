import os
import re
import cv2
import glob
import sys
import shutil
import json
import numpy as np


# src: [TN, TP, FN, FP]
def eva_indicators(src, TP_roi0=-1):
	TN = src[0]
	TP = src[1]
	FN = src[2]
	FP = src[3]
	if TP_roi0 < 0:
		TP_roi = TP
	else:
		TP_roi = TP_roi0

	precision = 0.0
	recall = 0.0
	Accuracy = 0.0
	F1Score = 0.0
	if TP + FP > 0:
		precision = TP_roi / (TP + FP)
	if FN + TP > 0:
		recall = TP_roi / (FN + TP)
	if TP + TN + FP + FN > 0:
		Accuracy = (TP_roi + TN) / (TP + TN + FP + FN)
	if precision + recall > 0:
		F1Score = 2 * precision * recall / (precision + recall)   
	return [precision, recall, Accuracy, F1Score]


def module_process(infiles, outfile, workdir="", cfg={}):
	src = np.zeros((3, 6))   # 3: [MD, OD, CS], 6: [Total, TN, TP, FN, FP, TP_roi]
	dst = np.zeros((5, 4))   # 5: [MD, MD_Roi, OD, OD_roi, CS], 4: [Precision, Recall, Accuracy, F1-Score]

	for infile in infiles:
		with open(infile) as f:
			fdata = json.load(f)

		src[0,0] += fdata["results"]["MD_Total"]
		src[0,1] += fdata["results"]["MD_TN"]
		src[0,2] += fdata["results"]["MD_TP"]
		src[0,3] += fdata["results"]["MD_FN"]
		src[0,4] += fdata["results"]["MD_FP"]
		src[0,5] += fdata["results"]["MD_TP_roi"]
		src[1,0] += fdata["results"]["OD_Total"]
		src[1,1] += fdata["results"]["OD_TN"]
		src[1,2] += fdata["results"]["OD_TP"]
		src[1,3] += fdata["results"]["OD_FN"]
		src[1,4] += fdata["results"]["OD_FP"]
		src[1,5] += fdata["results"]["OD_TP_roi"]
		src[2,0] += fdata["results"]["CS_Total"]
		src[2,1] += fdata["results"]["CS_TN"]
		src[2,2] += fdata["results"]["CS_TP"]
		src[2,3] += fdata["results"]["CS_FN"]
		src[2,4] += fdata["results"]["CS_FP"]

	dst[0, :] = eva_indicators(src[0,1:5])  # MD
	dst[1, :] = eva_indicators(src[0,1:5], src[0,5])  # MD-roi
	dst[2, :] = eva_indicators(src[1,1:5])  # OD
	dst[3, :] = eva_indicators(src[1,1:5], src[1,5])  # OD-roi
	dst[4, :] = eva_indicators(src[2,1:5])  # CS

	print("== Final QA result for list ==")
	print("MD: Total {0}, TN {1} TP {2} FN {3} FP {4}, TP_roi {5:.4f}".format(src[0,0], src[0,1], src[0,2], src[0,3], src[0,4], src[0,5]))
	print("MD: Precision {0:.4f}, Recall {1:.4f}, Accuracy {2:.4f}, F1-Score {3:.4f}".format(dst[0,0], dst[0,1], dst[0,2], dst[0,3]))
	print("MD ROI: Precision {0:.4f}, Recall {1:.4f}, Accuracy {2:.4f}, F1-Score {3:.4f}".format(dst[1,0], dst[1,1], dst[1,2], dst[1,3]))
	print("OD: Total {0}, TN {1} TP {2} FN {3} FP {4}, TP_roi {5:.4f}".format(src[1,0], src[1,1], src[1,2], src[1,3], src[1,4], src[1,5]))
	print("OD: Precision {0:.4f}, Recall {1:.4f}, Accuracy {2:.4f}, F1-Score {3:.4f}".format(dst[2,0], dst[2,1], dst[2,2], dst[2,3]))
	print("OD ROI: Precision {0:.4f}, Recall {1:.4f}, Accuracy {2:.4f}, F1-Score {3:.4f}".format(dst[3,0], dst[3,1], dst[3,2], dst[3,3]))
	print("CS: Total {0}, TN {1} TP {2} FN {3} FP {4}".format(src[2,0], src[2,1], src[2,2], src[2,3], src[2,4]))
	print("CS: Precision {0:.4f}, Recall {1:.4f}, Accuracy {2:.4f}, F1-Score {3:.4f}".format(dst[4,0], dst[4,1], dst[4,2], dst[4,3]))

	out_csv_file = os.path.join(outfile, "statistics.csv")
	with open(out_csv_file, 'w') as f:
		title_str_md = 'MD_Total,MD_TN,MD_TP,MD_FN,MD_FP,MD_TP_roi,MD_Precision,MD_Recall,MD_Accuracy,MD_F1Score,MD_Precision_roi,MD_Recall_roi,MD_Accuracy_roi,MD_F1Score_roi\n'
		data_str_md = '{0},{1},{2},{3},{4},{5:.4f},{6:.4f},{7:.4f},{8:.4f},{9:.4f},{10:.4f},{11:.4f},{12:.4f},{13:.4f}\n'.format(src[0,0], src[0,1], src[0,2], src[0,3], src[0,4], src[0,5], dst[0,0], dst[0,1], dst[0,2], dst[0,3], dst[1,0], dst[1,1], dst[1,2], dst[1,3])
		title_str_od = 'OD_Total,OD_TN,OD_TP,OD_FN,OD_FP,OD_TP_roi,OD_Precision,OD_Recall,OD_Accuracy,OD_F1Score,OD_Precision_roi,OD_Recall_roi,OD_Accuracy_roi,OD_F1Score_roi\n'
		data_str_od = '{0},{1},{2},{3},{4},{5:.4f},{6:.4f},{7:.4f},{8:.4f},{9:.4f},{10:.4f},{11:.4f},{12:.4f},{13:.4f}\n'.format(src[1,0], src[1,1], src[1,2], src[1,3], src[1,4], src[1,5], dst[2,0], dst[2,1], dst[2,2], dst[2,3], dst[3,0], dst[3,1], dst[3,2], dst[3,3])
		title_str_cs = 'CS_Total,CS_TN,CS_TP,CS_FN,CS_FP,,CS_Precision,CS_Recall,CS_Accuracy,CS_F1Score\n'
		data_str_cs = '{0},{1},{2},{3},{4},,{5:.4f},{6:.4f},{7:.4f},{8:.4f}\n'.format(src[2,0], src[2,1], src[2,2], src[2,3], src[2,4], dst[4,0], dst[4,1], dst[4,2], dst[4,3])
		f.write(title_str_md)
		f.write(data_str_md)
		f.write(title_str_od)
		f.write(data_str_od)
		f.write(title_str_cs)
		f.write(data_str_cs)

