import os
import sys
import copy
import json
import csv
import shutil
import numpy as np
import cv2
from cspp_od_mdbboxdist import cs_pp_one_frame_method1

np.set_printoptions(threshold=np.inf)
cfg_od_confidence_thre_daytime = 0.5
cfg_od_confidence_thre_lowlight = 0.3
cfg_od_overlap_thre = 0.4

cfg_font = cv2.FONT_HERSHEY_PLAIN
cfg_color00 = (0, 255, 255)  # color of dzone and bitmap for event=0
cfg_color01 = (0, 64, 255)   # color of dzone and bitmap for event=1
cfg_color1 = (255, 0, 0)     # color of md bbox
cfg_color2_cur = (0, 255, 128)    # color of od bbox and confidence
cfg_color2_real = (0, 255, 0)    # color of od bbox and confidence
cfg_colors_r = np.array(range(256)).reshape([-1, 1])
cfg_colors_0 = np.zeros(256).reshape([-1, 1])
cfg_colors3 = np.concatenate((cfg_colors_0, cfg_colors_0, cfg_colors_r), axis=1)  # color of distance point in md


def load_od_result(od_file, frm_num, od_w, od_h, yuv_w, yuv_h, dbg_en, workdir):
    with open(od_file, 'r') as f:
        od_text = f.read()
    f.close()

    # load od bboxes
    od_dict = json.loads(od_text)
    od_objects_list = od_dict['results']['objects']
    od_orig = []
    for frm_idx in range(frm_num):
        od_orig.append([0]) # objNum, [confidence, left, top, width, height], ...
    for obj in od_objects_list[1:]:
        frm_idx = obj['frame']   # start from 0
        od_orig[frm_idx].append([obj['confidence'], obj['left'], obj['top'], obj['width'], obj['height']])
        od_orig[frm_idx][0] += 1

    # remap to md frame size
    od_result = []
    for frm_idx in range(frm_num):
        cur_frm_orig = od_orig[frm_idx]
        obj_num = cur_frm_orig[0]
        cur_frm_result = [obj_num] # objNum, [confidence, left, top, width, height], ...
        for k in range(obj_num):
            sx = (cur_frm_orig[k+1][1] * yuv_w + (od_w >> 1)) // od_w
            sy = (cur_frm_orig[k+1][2] * yuv_h + (od_h >> 1)) // od_h
            w = (cur_frm_orig[k+1][3] * yuv_w + (od_w >> 1)) // od_w
            h = (cur_frm_orig[k+1][4] * yuv_h + (od_h >> 1)) // od_h
            cur_frm_result.append([cur_frm_orig[k+1][0], sx, sy, w, h])
        od_result.append(cur_frm_result)

    # check bboxes in od frames
    if dbg_en:
        od_path = os.path.dirname(od_file)
        seqname = os.path.splitext(os.path.basename(od_file))[0]
        od_mp4_path = os.path.join(od_path, seqname + ".mp4")
        od_yuv_in_path = os.path.join(workdir, seqname+"_in_"+str(od_w)+'x'+str(od_h)+'_P420.yuv')
        od_yuv_chk_path = os.path.join(workdir, seqname+"_check_"+str(od_w)+'x'+str(od_h)+'_P420.yuv')
        cmd="ffmpeg -pix_fmt yuv420p {0} -i {1}".format(od_yuv_in_path, od_mp4_path)
        print(cmd)
        os.system(cmd)
        color = cfg_color3[255]
        fin=open(od_yuv_in_path, mode="rb")
        fout=open(od_yuv_chk_path, mode="wb")
        for frm_idx in range(frm_num):
            b=fin.read(od_w * od_h * 3 // 2)
            if not b:
                break
            data_420p = np.frombuffer(b, dtype=np.uint8).reshape((od_h * 3 // 2, od_w))
            img_bgr = cv2.cvtColor(data_420p, cv2.COLOR_YUV420p2BGR)

            cur_frm_orig = od_orig[frm_idx]
            obj_num = cur_frm_orig[0]
            for k in range(obj_num):
                sx = cur_frm_orig[k+1][1]
                sy = cur_frm_orig[k+1][2]
                w = cur_frm_orig[k+1][3]
                h = cur_frm_orig[k+1][4]
                cv2.rectangle(img_bgr, (sx, sy), (sx + w, sy + h), color, 2)

            img_420p = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YUV_I420)
            fout.write(img_420p)

        fin.close()
        fout.close()

    return od_result, od_objects_list


def load_md_result(md_file, frm_num):
    with open(md_file, 'r') as f:
        md_text = f.read()
    f.close()

    md_dict = json.loads(md_text)
    md_objects_list = None
    md_result = []
    for frm_idx in range(frm_num):
        md_result.append([0]) # objNum, [left, top, width, height]

    if 'results' in md_dict.keys():
        if 'objects' in md_dict['results'].keys():
            md_objects_list = md_dict['results']['objects']
            for obj in md_objects_list:
                frm_idx = obj['frame']   # start from 0
                md_result[frm_idx].append([obj['left'], obj['top'], obj['width'], obj['height']])
                md_result[frm_idx][0] += 1

    if "distzone" in md_dict.keys():
        dzones = md_dict["distzone"]
    else:
        dzones = None

    gain = 0x10
    if "isp" in md_dict.keys():
        if "gain" in md_dict["isp"].keys():
            gain = int(md_dict["isp"]["gain"], 16)

    return [md_result, md_objects_list, dzones, gain]


# fnum: start from 0
def load_md_dist(md_dist_path, fnum):
    md_full_list = []
    dist_file = os.path.join(md_dist_path, f"{fnum + 1:0>4}_Full.csv")  # start fram 1

    with open(dist_file, 'r', newline='\r\n') as file:
        csv_reader = csv.reader(file, delimiter='\t')
        for row in csv_reader:
            md_full_list.append(row)
    file.close()

    item_num = len(md_full_list) - 1
    blk_h = int(md_full_list[item_num][0]) + 1
    blk_w = int(md_full_list[item_num][1]) + 1
    md_full = np.asarray(md_full_list[1:], dtype=np.int)
    dist = md_full[:, -2].astype(np.uint8).reshape((blk_h, blk_w))
    mask = md_full[:, -1].astype(np.uint8).reshape((blk_h, blk_w))

    return [dist, mask]


def load_bitmap(bitmap_file, w, h):
    if os.path.exists(bitmap_file):
        fp=open(bitmap_file, mode="rb")
        data_len = (w * h + 7) // 8
        b=fp.read(data_len)
        data = np.frombuffer(b, dtype=np.uint8)
        unpacked_data = np.unpackbits(data)
        bitmap = unpacked_data[0 : h*w].reshape((h, w))
        return bitmap
    else:
        return np.ones((h, w), dtype=np.uint8)


def module_process(infile, outfile, workdir="", cfg={}):
    seqname = os.path.splitext(os.path.basename(infile))[0]
    outdir = os.path.dirname(outfile)
    outfile = os.path.join(outdir, seqname + ".json")

    if "force" in cfg and cfg["force"] == 0:
        if os.path.isfile(outfile):
            print("[INFO] skip CS_PP because output file exist")
            return outfile

    md_file = infile
    md_w = cfg["in_width"]
    md_h = cfg["in_height"]
    md_path = os.path.dirname(infile)
    md_dist_path = os.path.join(md_path, seqname, "md_data")
    yuv_file = os.path.join(md_path, seqname+"_"+str(md_w)+"x"+str(md_h)+"_P422.yuv")
    od_file = cfg["in1_file"]
    od_w = cfg["in1_width"]
    od_h = cfg["in1_height"]
    dbg_yuv = os.path.join(workdir, seqname+"_"+str(md_w*3)+'x'+str(md_h)+'_P420.yuv')
    dbg_info = os.path.join(workdir, seqname+".txt")
    check_422to420_file = os.path.join(workdir, seqname+"_"+str(md_w)+'x'+str(md_h)+'_P420.yuv')
    if "bitmap_for_alarm" in cfg["param"].keys() and cfg["param"]["bitmap_for_alarm"] == "0.5m":
        bitmap_file = os.path.join(md_path, seqname, "dz_bitmap_0.5m.bin")
    else:
        bitmap_file = os.path.join(md_path, seqname, "dz_bitmap_1m.bin")
    if "od_interval" in cfg["param"].keys():
        od_interval = cfg["param"]["od_interval"]  # 0: without delay, >0: frames delayed to get OD results
    else:
        od_interval = 0  # without delay

    fin=open(yuv_file, mode="rb")
    fdbg_yuv=open(dbg_yuv, mode="wb")
    fdbg_info=open(dbg_info, mode="w")
    #fp_check_422to420 = open(check_422to420_file, mode="wb")

    # get frame number
    cnt=0
    while True:
        b=fin.read(md_w * md_h * 2)
        if not b:
            break
        cnt+=1
    frame_num=cnt

    # load MD & OD result
    md_result, md_objects_list, dzones, gain = load_md_result(md_file, frame_num)
    od_result, od_objects_list = load_od_result(od_file, frame_num, od_w, od_h, md_w, md_h, False, workdir)
    if gain < 0xc0:
        confidence_thre = cfg_od_confidence_thre_daytime
    else:
        confidence_thre = cfg_od_confidence_thre_lowlight

    #load bitmap
    md_dist_frm0 = load_md_dist(md_dist_path, 0)
    blk_w = md_dist_frm0[0].shape[1]
    blk_h = md_dist_frm0[0].shape[0]
    blk_step = int(md_w / blk_w + 0.5)
    blk_step_y = int(md_h / blk_h + 0.5)
    if blk_step != blk_step_y:
        print("Error: blk_step_x should be equal to blk_step_y.")
        exit()
    bitmap = load_bitmap(bitmap_file, blk_w, blk_h)

    #process each frame
    cnt=0
    fin.seek(0, 0)
    sentry_result = []
    cfg_data = {
            "w": md_w,
            "h": md_h,
            "blk_step": blk_step,
            }
    od_data = {
            "bboxes_info": [],   # [objNum, [confidence, left, top, width, height], ...]
            "confidence_thre": confidence_thre,
            "overlap_thre": cfg_od_overlap_thre
            }
    dbg_data = {
            "fdbg": fdbg_info,
            "out_path_prefix": workdir
            }
    while True:
        b=fin.read(md_w * md_h * 2)
        if not b:
            break
        fdbg_info.write("################## frame " + str(cnt) + " ##################\n")
        md_dist, md_mask = load_md_dist(md_dist_path, cnt)
        # get event
        md_data = {
            "bbox_info": md_result[cnt],    # [objNum, [left, top, width, height]]
            "dist": md_dist,
            "mask": md_mask
            }
        if od_interval > 0:
            if cnt >= od_interval:
                od_data["bboxes_info"] = od_result[(cnt // od_interval - 1) * od_interval]
            else:
                od_data["bboxes_info"] = [0]
        else:
                od_data["bboxes_info"] = od_result[cnt]
        dbg_data["out_path_prefix"] = workdir+"/frm"+str(cnt)
        event = cs_pp_one_frame_method1(cfg_data, bitmap, md_data, od_data, dbg_data)
        if event == 1:
            color0 = cfg_color01
        else:
            color0 = cfg_color00
       
        # convert yuv422p to yuv420p
        data_422p = np.frombuffer(b, dtype=np.uint8).reshape((md_h * 4, md_w // 2))
        data_420p = np.zeros((md_h * 3, md_w // 2), dtype=np.uint8)
        data_420p[0 : md_h*2, :] = data_422p[0 : md_h*2, :]
        data_420p[md_h*2 : md_h*3, :] = data_422p[md_h*2 : md_h*4 : 2, :]
        data_420p = data_420p.reshape((md_h * 3 // 2, md_w))
        #fp_check_422to420.write(data_420p)

        # convert yuv420p to bgr
        data_bgr = cv2.cvtColor(data_420p, cv2.COLOR_YUV2BGR_I420)

        # draw distzone
        if dzones is not None:
            for dzone_name in dzones.keys():
                dzone = np.array(dzones[dzone_name], dtype=np.int32)
                pts = dzone.reshape((-1, 1, 2))
                cv2.polylines(data_bgr, [pts], isClosed=True, color=color0, thickness=2)

        # three
        img_bgr = np.zeros((md_h, md_w*3, 3), dtype=np.uint8)
        img_bgr[:, 0:md_w, :] = data_bgr[:, :, :]           # left: dist without mask, md rect, od bboxes
        img_bgr[:, md_w:md_w*2, :] = data_bgr[:, :, :]      # middle: dist with mask, md rect, od bboxes
        img_bgr[:, md_w*2:md_w*3, :] = data_bgr[:, :, :]    # cspp result

        # draw dist & mask
        rad = blk_step // 2 - 2
        for i in range(blk_h):
            for j in range(blk_w):
                val = md_dist[i, j]
                ptx = j * blk_step + blk_step//2
                pty = i * blk_step + blk_step//2
                if val > 0:
                    cv2.circle(img_bgr, (ptx, pty), rad, cfg_colors3[val], -1)
                if(md_mask[i, j] != 0):
                    cv2.circle(img_bgr, (ptx + md_w, pty), rad, cfg_colors3[val], -1)

        # draw bitmap
        rad = 1
        for i in range(blk_h):
            for j in range(blk_w):
                if bitmap[i, j] > 0:
                    ptx = j * blk_step + blk_step//2
                    pty = i * blk_step + blk_step//2
                    cv2.circle(img_bgr, (ptx + md_w * 2, pty), rad, color0, -1)

        #draw md bbox
        if(md_result[cnt][0] > 0):
            sx = md_result[cnt][1][0]
            sy = md_result[cnt][1][1]
            w = md_result[cnt][1][2]
            h = md_result[cnt][1][3]
            cv2.rectangle(img_bgr, (sx, sy), (sx + w, sy + h), cfg_color1, 2)
            cv2.rectangle(img_bgr, (sx + md_w, sy), (sx + w + md_w, sy + h), cfg_color1, 2)

        # draw od bboxes
        if(od_result[cnt][0] > 0):
            for i in range(1, od_result[cnt][0]+1):
                conf = od_result[cnt][i][0]
                sx = od_result[cnt][i][1]
                sy = od_result[cnt][i][2]
                w = od_result[cnt][i][3]
                h = od_result[cnt][i][4]
                cv2.rectangle(img_bgr, (sx, sy), (sx + w, sy + h), cfg_color2_cur, 2)
                cv2.rectangle(img_bgr, (sx + md_w, sy), (sx + w + md_w, sy + h), cfg_color2_cur, 2)
                text = "{:.3f}".format(conf)
                (text_width, text_height) = cv2.getTextSize(text, cfg_font, fontScale=1.5, thickness=2)[0]
                cv2.putText(img_bgr, text, (sx, sy + int(text_height*1.5)), cfg_font, 1.5, cfg_color2_cur, 2, cv2.LINE_AA)
                cv2.putText(img_bgr, text, (sx + md_w, sy + int(text_height*1.5)), cfg_font, 1.5, cfg_color2_cur, 2, cv2.LINE_AA)
        if od_interval > 0 and cnt >= od_interval:
            od_list_real = od_result[(cnt // od_interval - 1) * od_interval]
            if(od_list_real[0] > 0):
                for i in range(1, od_list_real[0]+1):
                    conf = od_list_real[i][0]
                    sx = od_list_real[i][1]
                    sy = od_list_real[i][2]
                    w = od_list_real[i][3]
                    h = od_list_real[i][4]
                    cv2.rectangle(img_bgr, (sx, sy), (sx + w, sy + h), cfg_color2_real, 2)
                    cv2.rectangle(img_bgr, (sx + md_w, sy), (sx + w + md_w, sy + h), cfg_color2_real, 2)
                    text = "real: {:.3f}".format(conf)
                    (text_width, text_height) = cv2.getTextSize(text, cfg_font, fontScale=1.5, thickness=2)[0]
                    cv2.putText(img_bgr, text, (sx, sy + int(text_height*1.5)), cfg_font, 1.5, cfg_color2_real, 2, cv2.LINE_AA)
                    cv2.putText(img_bgr, text, (sx + md_w, sy + int(text_height*1.5)), cfg_font, 1.5, cfg_color2_real, 2, cv2.LINE_AA)

        # show cspp result in dbg yuv and save dbg image
        text = "Event: {:d}".format(event)
        (text_width, text_height) = cv2.getTextSize(text, cfg_font, fontScale=1.5, thickness=2)[0]
        cv2.putText(img_bgr, text, (md_w * 2, int(text_height*1.5)), cfg_font, 1.5, color0, 2, cv2.LINE_AA)
        img_420p = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YUV_I420)
        fdbg_yuv.write(img_420p)

        # save post process result to json file
        if (event):
            cur_result = {
                    "frame": cnt,
                    "event": 1,
                    }
            sentry_result.append(cur_result)

        cnt+=1

    fin.close()
    fdbg_yuv.close()
    fdbg_info.close()
    #fp_check_422to420.close()

    if os.path.isfile(infile):
        shutil.copyfile(infile, outfile)
        with open(outfile) as f:
            outfdata = json.load(f)
        outfdata["results"] = {
                "cs": sentry_result
                }
        if od_objects_list is not None:
            od_list = [od_objects_list[0], od_w, od_h] + od_objects_list[1:]
            outfdata["results"]["od_objects"] = od_list
        if md_objects_list is not None:
            outfdata["results"]["md_objects"] = md_objects_list
        with open(outfile, "w") as f:
            json.dump(outfdata, f, indent=4)

    return outfile


