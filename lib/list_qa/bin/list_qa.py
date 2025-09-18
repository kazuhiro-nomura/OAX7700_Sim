import os
import re
import cv2
import glob
import sys
import shutil
import json

def module_process(infiles, outfile, workdir="", cfg={}):
	TP = 0
	TN = 0
	FP = 0
	FN = 0
	MD_TN = 0
	MD_TP = 0
	MD_FN = 0
	MD_FP = 0
	MD_TP_roi = 0
	max_frames = 0

	for infile in infiles:
		with open(infile) as f:
			fdata = json.load(f)

		max_frames += fdata["results"]["Total"]
		MD_TN += fdata["results"]["MD_TN"]
		MD_TP += fdata["results"]["MD_TP"]
		MD_FN += fdata["results"]["MD_FN"]
		MD_FP += fdata["results"]["MD_FP"]
		MD_TP_roi += fdata["results"]["MD_TP_roi"]
		if "TN" in fdata["results"]:
			TN += fdata["results"]["TN"]
			TP += fdata["results"]["TP"]
			FN += fdata["results"]["FN"]
			FP += fdata["results"]["FP"]

	MD_precision = 0.0
	MD_recall = 0.0
	MD_Accuracy = 0.0
	MD_F1Score = 0.0
	if MD_TP + MD_FP > 0:
		MD_precision = MD_TP / (MD_TP + MD_FP)
	if MD_FN + MD_TP > 0:
		MD_recall = MD_TP / (MD_FN + MD_TP)
	if MD_TP + MD_TN + MD_FP + MD_FN > 0:
		MD_Accuracy = (MD_TP + MD_TN) / (MD_TP + MD_TN + MD_FP + MD_FN)
	if MD_precision + MD_recall > 0:
		MD_F1Score = 2 * MD_precision * MD_recall / (MD_precision + MD_recall)
	MD_precision_roi = 0.0
	MD_recall_roi = 0.0
	MD_Accuracy_roi = 0.0
	MD_F1Score_roi = 0.0
	if MD_TP_roi + MD_FP > 0:
		MD_precision_roi = MD_TP_roi / (MD_TP + MD_FP)
	if MD_FN + MD_TP_roi > 0:
		MD_recall_roi = MD_TP_roi / (MD_FN + MD_TP)
	if MD_TP + MD_TN + MD_FP + MD_FN > 0:
		MD_Accuracy_roi = (MD_TP_roi + MD_TN) / (MD_TP + MD_TN + MD_FP + MD_FN)
	if MD_precision_roi + MD_recall_roi > 0:
		MD_F1Score_roi = 2 * MD_precision_roi * MD_recall_roi / (MD_precision_roi + MD_recall_roi)

	precision = 0.0
	recall = 0.0
	accuracy = 0.0
	F1Score = 0.0
	if TP + FP > 0:
		precision = TP / (TP + FP)
	if FN + TP > 0:
		recall = TP / (FN + TP)
	if TP + TN + FP + FN > 0:
		accuracy = (TP + TN) / (TP + TN + FP + FN)
	if precision + recall > 0:
		F1Score = 2 * precision * recall / (precision + recall)

	print("== Final QA result for list ==")
	print("MD: Total {0}, TN {1} TP {2} FN {3} FP {4}, TP_roi {5:.4f}".format(max_frames, MD_TN, MD_TP, MD_FN, MD_FP, MD_TP_roi))
	print("MD: Precision {0:.4f}, Recall {1:.4f}, Accuracy {2:.4f}, F1-Score {3:.4f}".format(MD_precision, MD_recall, MD_Accuracy, MD_F1Score))
	print("MD ROI: Precision {0:.4f}, Recall {1:.4f}, Accuracy {2:.4f}, F1-Score {3:.4f}".format(MD_precision_roi, MD_recall_roi, MD_Accuracy_roi, MD_F1Score_roi))
	if TP + FP + FN + TP > 0:
		print("OC: Total {0}, TN {1} TP {2} FN {3} FP {4}".format(max_frames, TN, TP, FN, FP))
		print("OC: Precision {0:.4f}, Recall {1:.4f}, Accuracy {2:.4f}, F1-Score {3:.4f}".format(precision, recall, accuracy, F1Score))
