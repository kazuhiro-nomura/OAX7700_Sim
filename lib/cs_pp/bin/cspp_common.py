import numpy as np


def check_od_bbox(od_bboxes, confidence_thre, overlap_thre, fdbg_info):
    result_bboxes = [0]
    if od_bboxes[0] > 0:
        fdbg_info.write("### check_od_bbox ### input: confidence_thre=" + str(confidence_thre) + ', overlap_thre=' + str(overlap_thre) + '\n')
        fdbg_info.write("### check_od_bbox ### input: " + str(od_bboxes) + '\n')
        # remove od bboxes with (confidence < confidence_thre)
        tmp_bboxes = [0]
        for i in range(od_bboxes[0]):
            if od_bboxes[i+1][0] >= confidence_thre:
                tmp_bboxes.append(od_bboxes[i+1])
                tmp_bboxes[0] += 1
        # If two bboxes overlap, remove the one with smaller confidence.
        fdbg_info.write("### check_od_bbox ### tmp_bboxes: " + str(tmp_bboxes) + '\n')
        if tmp_bboxes[0] > 1:
            od_flags = np.ones(tmp_bboxes[0], dtype=np.uint8)
            for i in range(tmp_bboxes[0]):
                for j in range(tmp_bboxes[0]):
                    if i != j and od_flags[i] == 1 and od_flags[j] == 1:
                        overlap_x = min(tmp_bboxes[i+1][3] + tmp_bboxes[i+1][1], tmp_bboxes[j+1][3] + tmp_bboxes[j+1][1]) - max(tmp_bboxes[i+1][1], tmp_bboxes[j+1][1])
                        overlap_y = min(tmp_bboxes[i+1][4] + tmp_bboxes[i+1][2], tmp_bboxes[j+1][4] + tmp_bboxes[j+1][2]) - max(tmp_bboxes[i+1][2], tmp_bboxes[j+1][2])
                        if overlap_x > 0 and overlap_y > 0:
                            overlap = overlap_x * overlap_y
                            area_min = min(tmp_bboxes[i+1][3] * tmp_bboxes[i+1][4], tmp_bboxes[j+1][3] * tmp_bboxes[j+1][4])
                            if overlap > area_min * overlap_thre:
                                if tmp_bboxes[i+1][0] > tmp_bboxes[j+1][0]:
                                    od_flags[j] = 0
                                else:
                                    od_flags[i] = 0
            for i in range(tmp_bboxes[0]):
                if od_flags[i] > 0:
                    result_bboxes.append(tmp_bboxes[i+1])
                    result_bboxes[0] += 1
        else:
            result_bboxes = tmp_bboxes
    fdbg_info.write("### check_od_bbox ### result_bboxes: " + str(result_bboxes) + '\n')
    return result_bboxes


# rc: [sx, sy, ex, ey] in bitamp
# If step=8, then 4~11=>0, 12~19=>1, ... 
def bbox_to_rc(bbox, w, h, step, fdbg_info):
    min_val = 0
    max_val_x = w - 1 - step
    max_val_y = h - 1 - step
    sx = max(min_val, min(max_val_x, bbox[0] - step // 2)) // step
    sy = max(min_val, min(max_val_y, bbox[1] - step // 2)) // step
    ex = max(min_val, min(max_val_x, bbox[0] + bbox[2] - step // 2)) // step
    ey = max(min_val, min(max_val_y, bbox[1] + bbox[3] - step // 2)) // step
    fdbg_info.write("### bbox_to_rc ### " + str(bbox) + ", " + str([sx, sy, ex, ey]) + '\n')
    return [sx, sy, ex, ey]


