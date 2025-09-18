import os
import sys
import copy
import json
import shutil
import numpy as np
import cv2

cfg_OD_IoU_Thre = 0.5
cfg_bbox_dzone_intersect_thre = 16  # in pixels

cfg_font = cv2.FONT_HERSHEY_PLAIN
cfg_color_event0 = (0, 255, 255)  # color of dzone and bitmap for event=0
cfg_color_event1 = (0, 128, 255)   # color of dzone and bitmap for event=1
cfg_color_md = (255, 0, 0)     # color of md bbox
cfg_color_od_cur = (0, 255, 128)    # color of od bbox and confidence
cfg_color_od_real = (0, 255, 0)    # color of od bbox and confidence
cfg_color_golden = (0, 0, 255)  # color of golden data


# return value: out_obj_list,
# For each frame, there is an item in out_obj_list. The item is of "list" type. [obj_num, {obj0}, {obj1}, ...]
def load_objects(in_obj_list, frame_num, in_w, in_h, yuv_w, yuv_h):
    out_obj_list = []
    for i in range(frame_num):
        out_obj_list.append([0])
    if in_obj_list is not None:
        for obj in in_obj_list:
            frm_idx = obj['frame']   # frame index start from 0
            if frm_idx < frame_num:
                cur_obj = obj.copy()
                cur_obj['left'] = (obj['left'] * yuv_w + (in_w >> 1)) // in_w
                cur_obj['top'] = (obj['top'] * yuv_h + (in_h >> 1)) // in_h
                cur_obj['width'] = (obj['width'] * yuv_w + (in_w >> 1)) // in_w
                cur_obj['height'] = (obj['height'] * yuv_h + (in_h >> 1)) // in_h
                out_obj_list[frm_idx].append(cur_obj)
                out_obj_list[frm_idx][0] += 1
    return out_obj_list


# return value: out_cs_list,
# For each frame, there is an item in out_cs_list. The item is of int type.
def load_cs(in_cs_list, frame_num):
    out_cs_list = [0] * frame_num
    if in_cs_list is not None:
        for cs in in_cs_list:
            frm_idx = cs['frame']   # start from 0
            if frm_idx < frame_num and cs['event'] == 1:
                out_cs_list[frm_idx] = 1
    return out_cs_list


def load_golden_data(golden_json_file, frame_num, yuv_w, yuv_h):
    data_objects = None
    data_cs = None
    dzones = None
    with open(golden_json_file) as f:
        fdata = json.load(f)

    if 'objects' in fdata.keys():
        data_objects = fdata['objects']
    golden_objects = load_objects(data_objects, frame_num, yuv_w, yuv_h, yuv_w, yuv_h)
    if 'cs' in fdata.keys():
        data_cs = fdata['cs']
    golden_cs = load_cs(data_cs, frame_num)
    if "distzone" in fdata.keys():
        dzones = fdata["distzone"]
    return golden_objects, golden_cs, dzones


def load_detection_result(in_file, frame_num, yuv_w, yuv_h):
    md_w = 0
    md_h = 0
    od_w = 0
    od_h = 0
    data_md = None
    data_od = None
    data_cs = None
    with open(in_file) as f:
        fdata = json.load(f)

    if 'results' in fdata.keys() and 'md_objects' in fdata['results'].keys():
        data_md = fdata['results']['md_objects']
        md_w = fdata['width']
        md_h = fdata['height']
    md_objects = load_objects(data_md, frame_num, md_w, md_h, yuv_w, yuv_h)
    if 'results' in fdata.keys() and 'od_objects' in fdata['results'].keys():
        data_od = fdata['results']['od_objects'][3:]
        od_w = fdata['results']['od_objects'][1]
        od_h = fdata['results']['od_objects'][2]
    od_objects = load_objects(data_od, frame_num, od_w, od_h, yuv_w, yuv_h)
    if 'results' in fdata.keys() and 'cs' in fdata['results'].keys():
        data_cs = fdata['results']['cs']
    cs_result = load_cs(data_cs, frame_num)
    return md_objects, od_objects, cs_result


def check_bbox_in_dzone(bbox, bitmap):
    sx = bbox[0]
    ex = bbox[0] + bbox[2]
    sy = bbox[1]
    ey = bbox[1] + bbox[3]
    mask = np.zeros(bitmap.shape, bitmap.dtype)
    mask[sy:ey, sx:ex] = 1
    mask = mask & bitmap
    pnt_num = np.count_nonzero(mask)
    if pnt_num >= cfg_bbox_dzone_intersect_thre:
        return True
    return False


def ShowCsResults(data_422p, yuv_w, yuv_h, dzones, golden_obj_list, golden_cs, md_obj_list, od_obj_list_cur, od_obj_list_real, cs_event, cs_str):
    # convert yuv422p to yuv420p
    data_420p = np.zeros((yuv_h * 3, yuv_w // 2), dtype=np.uint8)
    data_420p[0 : yuv_h*2, :] = data_422p[0 : yuv_h*2, :]
    data_420p[yuv_h*2 : yuv_h*3, :] = data_422p[yuv_h*2 : yuv_h*4 : 2, :]
    data_420p = data_420p.reshape((yuv_h * 3 // 2, yuv_w))
    #fp_check_422to420.write(data_420p)

    # convert yuv420p to bgr
    img_bgr = cv2.cvtColor(data_420p, cv2.COLOR_YUV2BGR_I420)

    # draw distzone
    if cs_event == 1:
        color_event = cfg_color_event1
    else:
        color_event = cfg_color_event0
    if dzones is not None:
        for dzone_name in dzones.keys():
            dzone = np.array(dzones[dzone_name], dtype=np.int32)
            pts = dzone.reshape((-1, 1, 2))
            cv2.polylines(img_bgr, [pts], isClosed=True, color=color_event, thickness=2)

    # draw golden data
    if golden_obj_list[0] > 0:
        for obj in golden_obj_list[1:]:
            cv2.rectangle(img_bgr, (obj['left'], obj['top']), (obj['left'] + obj['width'], obj['top'] + obj['height']), cfg_color_golden, 2)
    text = "Event: {:d}".format(golden_cs)
    (text_width, text_height) = cv2.getTextSize(text, cfg_font, fontScale=1.5, thickness=2)[0]
    cv2.putText(img_bgr, text, (0, int(text_height*1.5)), cfg_font, 1.5, cfg_color_golden, 2, cv2.LINE_AA)

    #draw md bbox
    if md_obj_list[0] > 0:
        for obj in md_obj_list[1:]:
            cv2.rectangle(img_bgr, (obj['left'], obj['top']), (obj['left'] + obj['width'], obj['top'] + obj['height']), cfg_color_md, 2)

    # draw od bboxes
    if od_obj_list_cur[0] > 0:
        for obj in od_obj_list_cur[1:]:
            cv2.rectangle(img_bgr, (obj['left'], obj['top']), (obj['left'] + obj['width'], obj['top'] + obj['height']), cfg_color_od_cur, 2)
            text = "{:.3f}".format(obj['confidence'])
            (text_width, text_height) = cv2.getTextSize(text, cfg_font, fontScale=1.5, thickness=2)[0]
            cv2.putText(img_bgr, text, (obj['left'], obj['top'] + int(text_height*1.5)), cfg_font, 1.5, cfg_color_od_cur, 2, cv2.LINE_AA)
    if od_obj_list_real[0] > 0:
        for obj in od_obj_list_real[1:]:
            cv2.rectangle(img_bgr, (obj['left'], obj['top']), (obj['left'] + obj['width'], obj['top'] + obj['height']), cfg_color_od_real, 2)
            text = "real: {:.3f}".format(obj['confidence'])
            (text_width, text_height) = cv2.getTextSize(text, cfg_font, fontScale=1.5, thickness=2)[0]
            cv2.putText(img_bgr, text, (obj['left'], obj['top'] + int(text_height*1.5)), cfg_font, 1.5, cfg_color_od_real, 2, cv2.LINE_AA)

    # show cspp result
    text = "Event: {:d} {:s}".format(cs_event, cs_str)
    (text_width, text_height) = cv2.getTextSize(text, cfg_font, fontScale=1.5, thickness=2)[0]
    cv2.putText(img_bgr, text, (0, int(text_height*3.0)), cfg_font, 1.5, color_event, 2, cv2.LINE_AA)

    # convert to 420p
    img_420p = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YUV_I420)
    return img_420p


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


def module_process(infile, outfile, workdir="", cfg={}):
    seqname = os.path.splitext(os.path.basename(infile))[0]
    outdir = os.path.dirname(outfile)
    outfile = os.path.join(outdir, seqname + ".json")

    if "force" in cfg.keys() and cfg["force"] == 0:
        if os.path.isfile(outfile):
            print("[INFO] skip CS_PP because output file exist")
            return outfile

    cspp_w = cfg["in_width"]
    cspp_h = cfg["in_height"]
    yuv_w = cfg["in1_width"]
    yuv_h = cfg["in1_height"]
    in_yuv_path = cfg["in1_file"]  # YUV422P
    in1_json_path = in_yuv_path + ".json"
    dbg_yuv_path = os.path.join(workdir, seqname + "_" + str(yuv_w) + "x" + str(yuv_h) + "_P420.yuv")  # YUV420P
    out_mp4_path = os.path.join(outdir, seqname + ".mp4")
    fin_yuv=open(in_yuv_path, mode="rb")
    fdbg_yuv=open(dbg_yuv_path, mode="wb")
    dbg_info_path = os.path.join(workdir, seqname + "_dbg.txt")
    fdbg_info=open(dbg_info_path, mode="w")

    # get frame number
    cnt=0
    while True:
        b=fin_yuv.read(yuv_w * yuv_h * 2)
        if not b:
            break
        cnt+=1
    frame_num=cnt

    # load data
    golden_objects, golden_cs, dzones = load_golden_data(in1_json_path, frame_num, yuv_w, yuv_h)
    md_objects, od_objects, cs_result = load_detection_result(infile, frame_num, yuv_w, yuv_h)
    if dzones is not None:
        if "dzone_for_alarm" in cfg["param"].keys() and cfg["param"]["dzone_for_alarm"] == "0.5m":
            dzone_alarm = dzones["0.5m"]
        else:
            dzone_alarm = dzones["1m"]
        bitmap = np.zeros((yuv_h, yuv_w), dtype=np.uint8)
        pts = np.array([dzone_alarm], dtype=np.int32)
        cv2.fillPoly(bitmap, pts, color=1)
    else:
        bitmap = np.ones((yuv_h, yuv_w), dtype=np.uint8)
    if "od_interval" in cfg["param"].keys():
        od_interval = cfg["param"]["od_interval"]  # 0: without delay, >0: frames delayed to get OD results
    else:
        od_interval = 0  # without delay
    if "tp_margin" in cfg["param"].keys():
        tp_margin = cfg["param"]["tp_margin"]
    else:
        tp_margin = 0

    # statistics
    OD_total = 0
    OD_TP = 0
    OD_TP_roi = 0.0
    OD_TN = 0
    OD_FP = 0
    OD_FN = 0
    MD_TP = 0
    MD_TP_roi = 0.0
    MD_TN = 0
    MD_FP = 0
    MD_FN = 0
    CS_TP = 0
    CS_TN = 0
    CS_FP = 0
    CS_FN = 0
    fin_yuv.seek(0, 0)
    for frm_idx in range(frame_num):
        #show MD/OD/CS results
        b=fin_yuv.read(yuv_w * yuv_h * 2)
        if not b:
            print("read yuv422p data error for frame", frm_idx)
            break
        data_422p = np.frombuffer(b, dtype=np.uint8).reshape((yuv_h * 4, yuv_w // 2))

        # ground truth
        golden_obj_list_cur_frame = []  # includes person, animal, moving car, etc.
        golden_person_list_cur_frame = []  # Currently, only person detection is supported in OD, so use person data for OD performance evaluation.
        if golden_objects[frm_idx][0] > 0:
            for obj in golden_objects[frm_idx][1:]:
                bbox = [obj['left'], obj['top'], obj['width'], obj['height']]
                if check_bbox_in_dzone(bbox, bitmap):  # intersects with dzone  # ??? if obj["dist"] != "far":
                    golden_obj_list_cur_frame.append(obj)
                    if obj['label'] == 'person':
                        golden_person_list_cur_frame.append(obj)

        # md, od, cspp result
        md_obj_cur_frame = None
        if md_objects[frm_idx][0] > 0:
            md_obj = md_objects[frm_idx][1]
            bbox = [md_obj['left'], md_obj['top'], md_obj['width'], md_obj['height']]
            if check_bbox_in_dzone(bbox, bitmap):  # intersects with dzone
                md_obj_cur_frame = md_obj
        od_obj_list_cur_frame = []
        if od_objects[frm_idx][0] > 0:
            for obj in od_objects[frm_idx][1:]:
                bbox = [obj['left'], obj['top'], obj['width'], obj['height']]
                if check_bbox_in_dzone(bbox, bitmap):  # intersects with dzone
                    od_obj_list_cur_frame.append(obj)

        # QA MD
        gt_obj_num = len(golden_obj_list_cur_frame)
        if gt_obj_num == 0:
            # false event
            if md_obj_cur_frame is None:
                MD_TN += 1
            else:
                MD_FP += 1
        else:
            # true event
            if md_obj_cur_frame is None:
                MD_FN += 1
            else:
                MD_TP += 1
                # calc TP_ROI, use (A∩B)/(A∪B)
                # print("###### MD_TP_roi ######")
                # print("golden_obj:", golden_obj_list_cur_frame)
                # print("md:", md_obj_cur_frame)
                overlap_total = 0.0
                union_total = md_obj_cur_frame["width"] * md_obj_cur_frame["height"]
                for golden_obj in golden_obj_list_cur_frame:
                    overlap_left = max(golden_obj["left"], md_obj_cur_frame["left"])
                    overlap_top = max(golden_obj["top"], md_obj_cur_frame["top"])
                    overlap_right = min(golden_obj["left"] + golden_obj["width"], md_obj_cur_frame["left"] + md_obj_cur_frame["width"])
                    overlap_bottom = min(golden_obj["top"] + golden_obj["height"], md_obj_cur_frame["top"] + md_obj_cur_frame["height"])
                    if overlap_right > overlap_left and overlap_bottom > overlap_top:
                        overlap_area = int((overlap_bottom - overlap_top) * (overlap_right - overlap_left))
                    else:
                        overlap_area = 0
                    overlap_total += overlap_area
                    union_total += golden_obj["width"] * golden_obj["height"]
                union_total -= overlap_total
                MD_TP_roi += overlap_total / union_total
                # print("overlap_total =", overlap_total, "union_total =", union_total, "MD_TP_roi =", MD_TP_roi) 

        # QA OD
        gt_obj_num = len(golden_person_list_cur_frame)
        pred_obj_num = len(od_obj_list_cur_frame)
        OD_total += max(1, gt_obj_num)
        if gt_obj_num == 0:
            if pred_obj_num == 0:
                OD_TN += 1
            else:
                OD_FP += pred_obj_num
        else:
            if pred_obj_num == 0:
                OD_FN += gt_obj_num
            else:
                # match the predicted objects with golden objects based on IoU
                # print("###### OD_TP_roi ######")
                # print("golden_obj:", golden_person_list_cur_frame)
                # print("od:", od_obj_list_cur_frame)
                ious = np.zeros((gt_obj_num, pred_obj_num))
                flags = np.zeros((gt_obj_num, pred_obj_num))
                for i in range(gt_obj_num):
                    golden_obj = golden_person_list_cur_frame[i]
                    for j in range(pred_obj_num):
                        pred_obj = od_obj_list_cur_frame[j]
                        overlap_left = max(golden_obj["left"], pred_obj["left"])
                        overlap_top = max(golden_obj["top"], pred_obj["top"])
                        overlap_right = min(golden_obj["left"] + golden_obj["width"], pred_obj["left"] + pred_obj["width"])
                        overlap_bottom = min(golden_obj["top"] + golden_obj["height"], pred_obj["top"] + pred_obj["height"])
                        if overlap_right > overlap_left and overlap_bottom > overlap_top:
                            overlap_area = int((overlap_bottom - overlap_top) * (overlap_right - overlap_left))
                        else:
                            overlap_area = 0
                        union_area = golden_obj["width"] * golden_obj["height"] + pred_obj["width"] * pred_obj["height"] - overlap_area
                        ious[i, j] = overlap_area / union_area
                # print("ious:", ious)
                ious_tmp = np.where(ious >= cfg_OD_IoU_Thre, ious, 0)
                while 1:
                    max_idx = np.argmax(ious_tmp)
                    max_ii = max_idx // pred_obj_num
                    max_jj = max_idx % pred_obj_num
                    if ious_tmp[max_ii, max_jj] < cfg_OD_IoU_Thre:
                        break
                    else:
                        flags[max_ii, max_jj] = 1
                        ious_tmp[max_ii, :] = 0
                        ious_tmp[:, max_jj] = 0
                # print("flags:", flags)
                for i in range(gt_obj_num):
                    max_jj = np.argmax(flags[i,:])
                    if flags[i, max_jj] > 0:
                        OD_TP += 1
                        OD_TP_roi += ious[i, max_jj]
                        # print("Golden", i, "match with pred", max_jj, ". Add TP_roi", ious[i, max_jj])
                    else:
                        OD_FN += 1
                        # print("Golden", i, "add FN")
                for j in range(pred_obj_num):
                    if not(np.any(flags[:,j])):
                        OD_FP += 1
                        # print("Pred", j, "add FP")

        # QA CS
        if golden_cs[frm_idx] == 0:
            # false event
            if cs_result[frm_idx] == 0:  # result is negative, groundtruth is negative
                cs_str = "TN"
                CS_TN += 1
            else: # result is positive, groundtruth is negative
                is_done = False
                start_frm = max(0, frm_idx - tp_margin)
                end_frm = min(frame_num, frm_idx + tp_margin + 1)
                for k in range(start_frm, end_frm):   # check (2*tp_margin+1) frames golden data
                    if golden_cs[k] == 1:
                        cs_str = "TP"
                        CS_TP += 1
                        is_done = True
                        break
                if is_done == False:
                    cs_str = "FP"
                    CS_FP += 1
        else:
            if cs_result[frm_idx] == 0:  # result is negative, groundtruth is positive
                is_done = False
                start_frm = max(0, frm_idx - tp_margin)
                end_frm = min(frame_num, frm_idx + tp_margin + 1)
                for k in range(start_frm, end_frm):   # check (2*tp_margin+1) frames result data
                    if cs_result[k] == 1:
                        cs_str = "TP"
                        CS_TP += 1
                        is_done = True
                        break
                if is_done == False:
                    cs_str = "FN"
                    CS_FN += 1
            else:   # result is positive, groundtruth is positive
                cs_str = "TP"
                CS_TP += 1

        # show cs result in video
        if od_interval > 0 and frm_idx >= od_interval:
            od_list_real = od_objects[(frm_idx // od_interval - 1) * od_interval]
        else:
            od_list_real = [0]
        img_420p = ShowCsResults(data_422p, yuv_w, yuv_h, dzones, golden_objects[frm_idx], golden_cs[frm_idx], md_objects[frm_idx], od_objects[frm_idx], od_list_real, cs_result[frm_idx], cs_str)
        fdbg_yuv.write(img_420p)
        fdbg_info.write("frame" + str(frm_idx) + ": " + cs_str + "\n")

    MD_precision, MD_recall, MD_Accuracy, MD_F1Score = eva_indicators([MD_TN, MD_TP, MD_FN, MD_FP])
    MD_precision_roi, MD_recall_roi, MD_Accuracy_roi, MD_F1Score_roi = eva_indicators([MD_TN, MD_TP, MD_FN, MD_FP], MD_TP_roi)
    OD_precision, OD_recall, OD_Accuracy, OD_F1Score = eva_indicators([OD_TN, OD_TP, OD_FN, OD_FP])
    OD_precision_roi, OD_recall_roi, OD_Accuracy_roi, OD_F1Score_roi = eva_indicators([OD_TN, OD_TP, OD_FN, OD_FP], OD_TP_roi)
    CS_precision, CS_recall, CS_Accuracy, CS_F1Score = eva_indicators([CS_TN, CS_TP, CS_FN, CS_FP])
    
    print("QA result saved to {0}".format(outfile))
    print("MD: Total {0}, TN {1} TP {2} FN {3} FP {4}, TP_roi {5:.4f}".format(frame_num, MD_TN, MD_TP, MD_FN, MD_FP, MD_TP_roi))
    print("MD: Precision {0:.4f}, Recall {1:.4f}, Accuracy {2:.4f}, F1-Score {3:.4f}".format(MD_precision, MD_recall, MD_Accuracy, MD_F1Score))
    print("MD ROI: Precision {0:.4f}, Recall {1:.4f}, Accuracy {2:.4f}, F1-Score {3:.4f}".format(MD_precision_roi, MD_recall_roi, MD_Accuracy_roi, MD_F1Score_roi))
    print("OD: Total {0}, TN {1} TP {2} FN {3} FP {4}, TP_roi {5:.4f}".format(OD_total, OD_TN, OD_TP, OD_FN, OD_FP, OD_TP_roi))
    print("OD: Precision {0:.4f}, Recall {1:.4f}, Accuracy {2:.4f}, F1-Score {3:.4f}".format(OD_precision, OD_recall, OD_Accuracy, OD_F1Score))
    print("OD ROI: Precision {0:.4f}, Recall {1:.4f}, Accuracy {2:.4f}, F1-Score {3:.4f}".format(OD_precision_roi, OD_recall_roi, OD_Accuracy_roi, OD_F1Score_roi))
    print("CS: Total {0}, TN {1} TP {2} FN {3} FP {4}".format(frame_num, CS_TN, CS_TP, CS_FN, CS_FP))
    print("CS: Precision {0:.4f}, Recall {1:.4f}, Accuracy {2:.4f}, F1-Score {3:.4f}".format(CS_precision, CS_recall, CS_Accuracy, CS_F1Score))
    
    with open(infile) as f:
        fdata = json.load(f)
    fdata["results"]["MD_Total"] = frame_num
    fdata["results"]["MD_TN"] = MD_TN
    fdata["results"]["MD_TP"] = MD_TP
    fdata["results"]["MD_FN"] = MD_FN
    fdata["results"]["MD_FP"] = MD_FP
    fdata["results"]["MD_TP_roi"] = MD_TP_roi
    fdata["results"]["MD_Precision"] = MD_precision
    fdata["results"]["MD_Recall"] = MD_recall
    fdata["results"]["MD_Accuracy"] = MD_Accuracy
    fdata["results"]["MD_F1Score"] = MD_F1Score
    fdata["results"]["MD_Precision_roi"] = MD_precision_roi
    fdata["results"]["MD_Recall_roi"] = MD_recall_roi
    fdata["results"]["MD_Accuracy_roi"] = MD_Accuracy_roi
    fdata["results"]["MD_F1Score_roi"] = MD_F1Score_roi
    fdata["results"]["OD_Total"] = OD_total
    fdata["results"]["OD_TN"] = OD_TN
    fdata["results"]["OD_TP"] = OD_TP
    fdata["results"]["OD_FN"] = OD_FN
    fdata["results"]["OD_FP"] = OD_FP
    fdata["results"]["OD_TP_roi"] = OD_TP_roi
    fdata["results"]["OD_Precision"] = OD_precision
    fdata["results"]["OD_Recall"] = OD_recall
    fdata["results"]["OD_Accuracy"] = OD_Accuracy
    fdata["results"]["OD_F1Score"] = OD_F1Score
    fdata["results"]["OD_Precision_roi"] = OD_precision_roi
    fdata["results"]["OD_Recall_roi"] = OD_recall_roi
    fdata["results"]["OD_Accuracy_roi"] = OD_Accuracy_roi
    fdata["results"]["OD_F1Score_roi"] = OD_F1Score_roi
    fdata["results"]["CS_Total"] = frame_num
    fdata["results"]["CS_TN"] = CS_TN
    fdata["results"]["CS_TP"] = CS_TP
    fdata["results"]["CS_FN"] = CS_FN
    fdata["results"]["CS_FP"] = CS_FP
    fdata["results"]["CS_Precision"] = CS_precision
    fdata["results"]["CS_Recall"] = CS_recall
    fdata["results"]["CS_Accuracy"] = CS_Accuracy
    fdata["results"]["CS_F1Score"] = CS_F1Score
    with open(outfile, "w") as f:
        json.dump(fdata, f, indent=4)

    fin_yuv.close()
    fdbg_yuv.close()
    fdbg_info.close()

    # save to mp4 file
    cmd = f"%s -f rawvideo -pix_fmt yuv420p -s:v %dx%d -i %s -c:v libx264 %s"%(cfg["ffmpeg_path"], yuv_w, yuv_h, dbg_yuv_path, out_mp4_path)
    os.system(cmd)
    os.system("rm %s"%(dbg_yuv_path))

    return outfile

