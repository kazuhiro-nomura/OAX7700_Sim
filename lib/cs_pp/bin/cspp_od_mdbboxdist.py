import numpy as np

from cspp_common import bbox_to_rc
from cspp_common import check_od_bbox

cfg_check_rcroi_trust_od_bbox = True
cfg_check_rcroi_expand_od = False
cfg_check_rcroi_pntnum_thre_od = 3
cfg_check_rcroi_trust_md_bbox = False
cfg_check_rcroi_expand_md = True
cfg_check_rcroi_pntnum_thre_md = 5
cfg_check_rcroi_dist_thre = 32
cfg_check_rcroi_point_check_range = 1
cfg_check_rcroi_dbg_en = False

#rc: [sx, sy, ex, ey] in bitmap
# expand rc in three directions: left, bottom, right
def check_rc_in_roi(rc, md_dist, md_mask, bitmap, trust_rc, expand_en, pnt_num_thre, fdbg_info, dbg_str):
    fdbg_info.write("### check_rc_in_roi ### " + str(rc) + ", trust_rc = " + str(trust_rc) + '\n')

    bitmap_w = bitmap.shape[1]
    bitmap_h = bitmap.shape[0]
    mask_thre = np.zeros(md_mask.shape, md_mask.dtype)
    mask_thre[:,:] = md_dist[:,:] > cfg_check_rcroi_dist_thre
    mask_thre[:,:] = mask_thre[:,:] & md_mask[:,:]

    mask_roi = np.zeros(md_mask.shape, md_mask.dtype)
    if trust_rc:
        mask_roi[rc[1]:rc[3]+1, rc[0]:rc[2]+1] = 1
    else:
        mask_roi[rc[1]:rc[3]+1, rc[0]:rc[2]+1] = mask_thre[rc[1]:rc[3]+1, rc[0]:rc[2]+1]
    if cfg_check_rcroi_dbg_en:
        np.savetxt(dbg_str+"_mask_roi_orig.csv", mask_roi, fmt='%1d', delimiter=' ')

    # rc
    mask_dst = np.zeros(md_mask.shape, md_mask.dtype)
    mask_dst[rc[1]:rc[3]+1, rc[0]:rc[2]+1] = mask_roi[rc[1]:rc[3]+1, rc[0]:rc[2]+1] & bitmap[rc[1]:rc[3]+1, rc[0]:rc[2]+1]
    pnt_num = np.count_nonzero(mask_dst)
    if pnt_num > pnt_num_thre:
        return True

    if expand_en:
        # expand left
        for i in range(rc[0]-1, 0, -1):
            for j in range(rc[3], rc[1], -1):
                if mask_thre[j, i]:
                    for k in range (1, cfg_check_rcroi_point_check_range+1):
                        if mask_roi[j, i+k]:
                            mask_roi[j, i] = 1
        if cfg_check_rcroi_dbg_en:
            np.savetxt(dbg_str+"_mask_roi_left.csv", mask_roi, fmt='%1d', delimiter=' ')
        mask_dst = np.zeros(md_mask.shape, md_mask.dtype)
        mask_dst[rc[1]:rc[3]+1, 0:rc[0]] = mask_roi[rc[1]:rc[3]+1, 0:rc[0]] & bitmap[rc[1]:rc[3]+1, 0:rc[0]]
        pnt_num_left = np.count_nonzero(mask_dst)
        pnt_num += pnt_num_left
        if pnt_num > pnt_num_thre:
            return True

        # expand right
        for i in range(rc[2], bitmap_w):
            for j in range(rc[3], rc[1], -1):
                if mask_thre[j, i]:
                    for k in range (1, cfg_check_rcroi_point_check_range+1):
                        if mask_roi[j, i-k]:
                            mask_roi[j, i] = 1
        if cfg_check_rcroi_dbg_en:
            np.savetxt(dbg_str+"_mask_roi_right.csv", mask_roi, fmt='%1d', delimiter=' ')
        mask_dst = np.zeros(md_mask.shape, md_mask.dtype)
        mask_dst[rc[1]:rc[3]+1, rc[2]+1:bitmap_w] = mask_roi[rc[1]:rc[3]+1, rc[2]+1:bitmap_w] & bitmap[rc[1]:rc[3]+1, rc[2]+1:bitmap_w]
        pnt_num_right = np.count_nonzero(mask_dst)
        pnt_num += pnt_num_right
        if pnt_num > pnt_num_thre:
            return True

        # expand bottom
        sx = max(0, rc[0] - cfg_check_rcroi_point_check_range)
        ex = min(bitmap_w, rc[2] + cfg_check_rcroi_point_check_range + 1)
        for j in range(rc[3]+1, bitmap_h):
            for i in range(sx, ex):
                if mask_thre[j, i]:
                    for k in range (1, cfg_check_rcroi_point_check_range+1):
                        if mask_roi[j-k, i]:
                            mask_roi[j, i] = 1
        if cfg_check_rcroi_dbg_en:
            np.savetxt(dbg_str+"_mask_roi_bottom.csv", mask_roi, fmt='%1d', delimiter=' ')
        mask_dst = np.zeros(md_mask.shape, md_mask.dtype)
        mask_dst[rc[3]+1:bitmap_h, sx:ex] = mask_roi[rc[3]+1:bitmap_h, sx:ex] & bitmap[rc[3]+1:bitmap_h, sx:ex]
        pnt_num_bottom = np.count_nonzero(mask_dst)
        pnt_num += pnt_num_bottom
        if pnt_num > pnt_num_thre:
            return True

    return False


'''
    cfg_data: {"w", "h", "blk_step"}
    bitmap:  md_dist, md_mask, bitmap are of the same size
    md_data: {"bbox_info", "dist", "mask"}. # bbox_info: [objNum, [left, top, width, height]].
    od_data: {"bboxes_info", "confidence_thre", "overlap_thre"}. # bboxes_info: [objNum, [confidence, left, top, width, height], ...].
    dbg_data: {"fdbg", "out_path_prefix"}
'''
def cs_pp_one_frame_method1(cfg_data, bitmap, md_data, od_data, dbg_data):
    if md_data["bbox_info"][0] > 0:
        md_rc = bbox_to_rc(md_data["bbox_info"][1], cfg_data["w"], cfg_data["h"], cfg_data["blk_step"], dbg_data["fdbg"])  #rc: [sx, sy, ex, ey] in bitmap
    else:
        md_rc = None

    # objects detected in od
    if od_data["bboxes_info"][0] > 0:
        od_bboxes = check_od_bbox(od_data["bboxes_info"], od_data["confidence_thre"], od_data["overlap_thre"], dbg_data["fdbg"])
        for i in range(od_bboxes[0]):
            od_rc = bbox_to_rc(od_bboxes[i+1][1:], cfg_data["w"], cfg_data["h"], cfg_data["blk_step"], dbg_data["fdbg"])  #rc: [sx, sy, ex, ey] in bitmap
            if check_rc_in_roi(od_rc, md_data["dist"], md_data["mask"], bitmap, cfg_check_rcroi_trust_od_bbox, cfg_check_rcroi_expand_od, cfg_check_rcroi_pntnum_thre_od, dbg_data["fdbg"], dbg_data["out_path_prefix"]+"_od"+str(i)):
                return 1

    # no object detected in od, consider md
    if md_rc is not None:
        if check_rc_in_roi(md_rc, md_data["dist"], md_data["mask"], bitmap, cfg_check_rcroi_trust_md_bbox, cfg_check_rcroi_expand_md, cfg_check_rcroi_pntnum_thre_md, dbg_data["fdbg"], dbg_data["out_path_prefix"]+"_md"):
            return 1

    return 0


