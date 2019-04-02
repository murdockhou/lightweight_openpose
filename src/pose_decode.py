"""
@author: autocyz
@contact: autocyz@163.com
@file: pose_decode.py
@function: get joints and limbs from CPM and PAF
@time: 18-10-20
"""

import cv2
import numpy as np
import math

from scipy.ndimage.filters import gaussian_filter, maximum_filter
from scipy.ndimage.morphology import generate_binary_structure

# limb连接的点的id
joint_to_limb_heatmap_relationship = [[0, 1], [1,2], [3, 4], [4, 5], [6, 7], [7, 8],
                                      [9, 10], [10, 11], [12, 13], [13, 0], [13, 3], [13, 6], [13, 9]]
# joint_to_limb_heatmap_relationship = [[1,0], [2,1], [4, 3], [5, 4], [7, 6], [8, 7],
#                                       [10, 9], [11, 10], [13, 12], [0, 13], [3, 13], [6, 13], [9, 13]]

# 每个limb在paf中的id,例如第一个limb就是前两个paf的channel
paf_xy_coords_per_limb = [[0, 1], [2, 3], [4, 5], [6, 7], [8, 9], [10, 11], [12, 13], [14, 15], [16, 17],
                          [18, 19], [20, 21], [22, 23], [24, 25]]

colors = [[255, 0, 0], [0,255,0], [0, 0, 255],
          [255, 0, 255], [0,255,255], [255, 0, 255],
          [100, 60, 39], [35, 200, 111], [200, 40, 145],
          [238, 248, 220], [255, 165, 100], [20, 205, 50],
          [0, 191, 255], [0, 0, 205], [220, 20, 60],
          [100, 20, 0]]

NUM_JOINTS = 14
NUM_LIMBS = len(joint_to_limb_heatmap_relationship)


def find_peaks(param, img):
    """
    Given a (grayscale) image, find local maxima whose value is above a given
    threshold (param['thre1'])
    :param img: Input image (2d array) where we want to find peaks
    :return: 2d np.array containing the [x,y] coordinates of each peak found
    in the image
    """
    peaks_binary = (maximum_filter(img, footprint=generate_binary_structure(
        2, 1)) == img) * (img > param['thre1'])
    # Note reverse ([::-1]): we return [[x y], [x y]...] instead of [[y x], [y
    # x]...]
    return np.array(np.nonzero(peaks_binary)[::-1]).T


def find_peaks_v2(param, img):
    dis_thres = 5
    x, y = np.where(img > param['thre1'])
    coordinate = list(zip(x, y))
    scores = []
    for coor in coordinate:
        scores.append(img[coor])
    s = np.asarray(scores)
    sIndex = s.argsort()[::-1]
    keepIndex = []
    while sIndex.size > 0:
        keepIndex.append(sIndex[0])
        sIndex = sIndex[1:]
        last = []
        for index in sIndex:
            dis = np.linalg.norm(np.asarray(coordinate[keepIndex[-1]]) - np.asarray(coordinate[index]))
            if dis > dis_thres:
                last.append(index)
        sIndex = np.asarray(last)
    peaks = []
    for index in keepIndex:
        coorX = coordinate[index][1]
        coorY = coordinate[index][0]
        peaks.append([coorX, coorY])
    return np.asarray(peaks)



def compute_resized_coords(coords, resizeFactor):
    """
    Given the index/coordinates of a cell in some input array (e.g. image),
    provides the new coordinates if that array was resized by making it
    resizeFactor times bigger.
    E.g.: image of size 3x3 is resized to 6x6 (resizeFactor=2), we'd like to
    know the new coordinates of cell [1,2] -> Function would return [2.5,4.5]
    :param coords: Coordinates (indices) of a cell in some input array
    :param resizeFactor: Resize coefficient = shape_dest/shape_source. E.g.:
    resizeFactor=2 means the destination array is twice as big as the
    original one
    :return: Coordinates in an array of size
    shape_dest=resizeFactor*shape_source, expressing the array indices of the
    closest point to 'coords' if an image of size shape_source was resized to
    shape_dest
    """

    # 1) Add 0.5 to coords to get coordinates of center of the pixel (e.g.
    # index [0,0] represents the pixel at location [0.5,0.5])
    # 2) Transform those coordinates to shape_dest, by multiplying by resizeFactor
    # 3) That number represents the location of the pixel center in the new array,
    # so subtract 0.5 to get coordinates of the array index/indices (revert
    # step 1)

    return (np.array(coords, dtype=float) + 0.5) * resizeFactor - 0.5


def NMS(param, heatmaps, upsampFactor=1., bool_refine_center=True, bool_gaussian_filt=False):
    """
    NonMaximaSuppression: find peaks (local maxima) in a set of grayscale images
    :param heatmaps: set of grayscale images on which to find local maxima (3d np.array,
    with dimensions image_height x image_width x num_heatmaps)
    :param upsampFactor: Size ratio between CPM heatmap output and the input image size.
    Eg: upsampFactor=16 if original image was 480x640 and heatmaps are 30x40xN
    :param bool_refine_center: Flag indicating whether:
     - False: Simply return the low-res peak found upscaled by upsampFactor (subject to grid-snap)
     - True: (Recommended, very accurate) Upsample a small patch around each low-res peak and
     fine-tune the location of the peak at the resolution of the original input image
    :param bool_gaussian_filt: Flag indicating whether to apply a 1d-GaussianFilter (smoothing)
    to each upsampled patch before fine-tuning the location of each peak.
    :return: a NUM_JOINTS x 5 np.array where each row represents a joint type (0=nose, 1=neck...)
    and the columns indicate the {x,y} position, the score (probability), a unique id (counter)
    and a flag this point used for assignment
    """
    # MODIFIED BY CARLOS: Instead of upsampling the heatmaps to heatmap_avg and
    # then performing NMS to find peaks, this step can be sped up by ~25-50x by:
    # (9-10ms [with GaussFilt] or 5-6ms [without GaussFilt] vs 250-280ms on RoG
    # 1. Perform NMS at (low-res) CPM's output resolution
    # 1.1. Find peaks using scipy.ndimage.filters.maximum_filter
    # 2. Once a peak is found, take a patch of 5x5 centered around the peak, upsample it, and
    # fine-tune the position of the actual maximum.
    #  '-> That's equivalent to having found the peak on heatmap_avg, but much faster because we only
    #      upsample and scan the 5x5 patch instead of the full (e.g.) 480x640

    joint_list_per_joint_type = []
    cnt_total_joints = 0

    # For every peak found, win_size specifies how many pixels in each
    # direction from the peak we take to obtain the patch that will be
    # upsampled. Eg: win_size=1 -> patch is 3x3; win_size=2 -> 5x5
    # (for BICUBIC interpolation to be accurate, win_size needs to be >=2!)
    win_size = 2

    for joint in range(NUM_JOINTS):
        map_orig = heatmaps[:, :, joint]
        # peak_coords = find_peaks(param, map_orig)
        peak_coords = find_peaks_v2(param, map_orig)
        peaks = np.zeros((len(peak_coords), 5))
        for i, peak in enumerate(peak_coords):
            if bool_refine_center:
                x_min, y_min = np.maximum(0, peak - win_size)
                x_max, y_max = np.minimum(
                    np.array(map_orig.T.shape) - 1, peak + win_size)

                # Take a small patch around each peak and only upsample that
                # tiny region
                patch = map_orig[y_min:y_max + 1, x_min:x_max + 1]
                map_upsamp = cv2.resize(
                    patch, None, fx=upsampFactor, fy=upsampFactor, interpolation=cv2.INTER_CUBIC)

                # Gaussian filtering takes an average of 0.8ms/peak (and there might be
                # more than one peak per joint!) -> For now, skip it (it's
                # accurate enough)
                map_upsamp = gaussian_filter(
                    map_upsamp, sigma=3) if bool_gaussian_filt else map_upsamp

                # Obtain the coordinates of the maximum value in the patch
                location_of_max = np.unravel_index(
                    map_upsamp.argmax(), map_upsamp.shape)
                # Remember that peaks indicates [x,y] -> need to reverse it for
                # [y,x]
                location_of_patch_center = compute_resized_coords(
                    peak[::-1] - [y_min, x_min], upsampFactor)
                # Calculate the offset wrt to the patch center where the actual
                # maximum is
                refined_center = (location_of_max - location_of_patch_center)
                peak_score = map_upsamp[location_of_max]
            else:
                refined_center = [0, 0]
                # Flip peak coordinates since they are [x,y] instead of [y,x]
                peak_score = map_orig[tuple(peak[::-1])]
            peaks[i, :] = tuple([int(math.floor(x)) for x in compute_resized_coords(
                peak_coords[i], upsampFactor) + refined_center[::-1]]) + (peak_score, cnt_total_joints, 0)
            cnt_total_joints += 1
        joint_list_per_joint_type.append(peaks)
    return joint_list_per_joint_type


def find_connected_joints(param, paf_upsamp, joint_list_per_joint_type, num_intermed_pts=10, max_dist_thresh=1,
                          max_paf_score_thresh=0.7):
    """
    For every type of limb (eg: forearm, shin, etc.), look for every potential
    pair of joints (eg: every wrist-elbow combination) and evaluate the PAFs to
    determine which pairs are indeed body limbs.
    :param paf_upsamp: PAFs upsampled to the original input image resolution
    :param joint_list_per_joint_type: See 'return' doc of NMS()
    :param num_intermed_pts: Int indicating how many intermediate points to take
    between joint_src and joint_dst, at which the PAFs will be evaluated
    :param max_dist_thresh: do not set limb connected if limb length is bigger than (this threshold * img_height / img_width)
    :return: List of NUM_LIMBS rows. For every limb_type (a row) we store
    a list of all limbs of that type found (eg: all the right forearms).
    For each limb (each item in connected_limbs[limb_type]), we store 5 cells:
    # {joint_src_id,joint_dst_id}: a unique number associated with each joint,
    # limb_score_penalizing_long_dist: a score of how good a connection
    of the joints is, penalized if the limb length is too long
    # {joint_src_index,joint_dst_index}: the index of the joint within
    all the joints of that type found (eg: the 3rd right elbow found)
    """
    connected_limbs = []
    long_dist_thresh = np.array([paf_upsamp.shape[1], paf_upsamp.shape[0]]) * max_dist_thresh
    # Auxiliary array to access paf_upsamp quickly
    # 这个limb_intermed_coords, 是个(4, num_intermed_pts)ndarray, 对其赋值完之后,
    # 第一行表示的取得两个关节点之间10个值的y坐标
    # 第二行表示的取得的两个关节点之间10个值的x坐标
    # 第三 四行表示这两个关节点的所在的paf channel位置,根据paf_xy_coords_per_limb得到
    limb_intermed_coords = np.empty((4, num_intermed_pts), dtype=np.intp)
    for limb_type in range(NUM_LIMBS):
        # 假如我们希望链接脖子到头的点，那么这一步就是找到所有的头和脖子点的过程
        # List of all joints of type A found, where A is specified by limb_type
        # (eg: a right forearm starts in a right elbow)
        joints_src = joint_list_per_joint_type[joint_to_limb_heatmap_relationship[limb_type][0]]
        # List of all joints of type B found, where B is specified by limb_type
        # (eg: a right forearm ends in a right wrist)
        joints_dst = joint_list_per_joint_type[joint_to_limb_heatmap_relationship[limb_type][1]]
        if len(joints_src) == 0 or len(joints_dst) == 0:
            # No limbs of this type found (eg: no right forearms found because
            # we didn't find any right wrists or right elbows)
            # print ('no limbs of this type found.')
            connected_limbs.append([])
        else:
            connections = np.empty((0, 5))
            # Specify the paf index that contains the x-coord of the paf for
            # this limb
            # 这一对paf在heatmap中对应的id

            limb_intermed_coords[2, :] = paf_xy_coords_per_limb[limb_type][0]
            # print (paf_xy_coords_per_limb[limb_type][0])
            # And the y-coord paf index
            limb_intermed_coords[3, :] = paf_xy_coords_per_limb[limb_type][1]
            for i, joint_src in enumerate(joints_src):
                # print (i, joint_src)
                # Try every possible joints_src[i]-joints_dst[j] pair and see
                # if it's a feasible limb
                best_score = 0.0
                best_connection = []
                for j, joint_dst in enumerate(joints_dst):
                    # print (j, joint_dst)
                    # Subtract the position of both joints to obtain the
                    # direction of the potential limb
                    limb_dir = joint_dst[:2] - joint_src[:2]
                    if (np.abs(limb_dir) > long_dist_thresh).any():
                        # print ('000')
                        continue
                    # Compute the distance/length of the potential limb (norm
                    # of limb_dir)
                    limb_dist = np.sqrt(np.sum(limb_dir ** 2)) + 1e-8
                    limb_dir = limb_dir / limb_dist  # Normalize limb_dir to be a unit vector

                    # Linearly distribute num_intermed_pts points from the x
                    # coordinate of joint_src to the x coordinate of joint_dst
                    limb_intermed_coords[1, :] = np.round(np.linspace(
                        joint_src[0], joint_dst[0], num=num_intermed_pts))
                    # Same for the y coordinate
                    limb_intermed_coords[0, :] = np.round(np.linspace(
                        joint_src[1], joint_dst[1], num=num_intermed_pts))

                    # 找到两个关节点之间的paf值
                    intermed_paf = paf_upsamp[limb_intermed_coords[0, :],
                                              limb_intermed_coords[1, :], limb_intermed_coords[2:4, :]].T

                    score_intermed_pts = intermed_paf.dot(limb_dir)
                    score_penalizing_long_dist = score_intermed_pts.mean()
                    # score_penalizing_long_dist = score_intermed_pts.mean(
                    # ) + min(0.5 * paf_upsamp.shape[0] / limb_dist - 1, 0)
                    # Criterion 1: At least 80% of the intermediate points have
                    # a score higher than thre2
                    criterion1 = (np.count_nonzero(
                        score_intermed_pts > param['thre2']) > 0.8 * num_intermed_pts)

                    # Criterion 2: Mean score, penalized for large limb
                    # distances (larger than half the image height), is
                    # positive
                    # print ('score penalizing long dist: {}, mean: {}'.format(score_penalizing_long_dist, score_intermed_pts.mean()))
                    criterion2 = (score_penalizing_long_dist > param['thre2'])
                    # print()
                    if criterion1 and criterion2 and score_penalizing_long_dist > best_score:
                        best_score = score_penalizing_long_dist
                        best_connection = [joint_src[3], joint_dst[3], score_penalizing_long_dist, i, j]
                        # print('best_connection')
                        # if best_score > max_paf_score_thresh:
                        #     break

                # Last value is the combined paf(+limb_dist) + heatmap
                # scores of both joints
                if best_connection:
                    # print(best_connection[2])
                    # print('111', connections)
                    connections = np.vstack([connections, np.array(best_connection)])

            connected_limbs.append(connections)

    return connected_limbs


def group_limbs_of_same_person(connected_limbs, joint_list):
    """
    Associate limbs belonging to the same person together.
    :param connected_limbs: See 'return' doc of find_connected_joints()
    :param joint_list: unravel'd version of joint_list_per_joint [See 'return' doc of NMS()]
    :return: 2d np.array of size num_people x (NUM_JOINTS+2). For each person found:
    # First NUM_JOINTS columns contain the index (in joint_list) of the joints associated
    with that person (or -1 if their i-th joint wasn't found)
    # 2nd-to-last column: Overall score of the joints+limbs that belong to this person
    # Last column: Total count of joints found for this person
    """

    # person_to_joint_assoc是一个列表的列表, 每个子列表都是一个1x(num_joints+2)的ndarray
    person_to_joint_assoc = []

    for limb_type in range(NUM_LIMBS):
        # 需要连接关节点的类型
        joint_src_type, joint_dst_type = joint_to_limb_heatmap_relationship[limb_type]

        # 对connect_limbs里对应的limb_type那行进行遍历, 就是遍历所有的这个类型的关节
        for limb_info in connected_limbs[limb_type]:

            person_assoc_idx = []
            # 如果在已有的person_to_joint_assoc里,已有的连接和此时需要处理的limb_info保存的unique_id一样,
            # ??????
            for person, person_limbs in enumerate(person_to_joint_assoc):
                if person_limbs[joint_src_type] == limb_info[0] or person_limbs[joint_dst_type] == limb_info[1]:
                    person_assoc_idx.append(person)

            # If one of the joints has been associated to a person, and either
            # the other joint is also associated with the same person or not
            # associated to anyone yet:
            if len(person_assoc_idx) == 1:
                person_limbs = person_to_joint_assoc[person_assoc_idx[0]]
                # If the other joint is not associated to anyone yet,
                if person_limbs[joint_dst_type] != limb_info[1]:
                    # Associate it with the current person
                    person_limbs[joint_dst_type] = limb_info[1]
                    # Increase the number of limbs associated to this person
                    person_limbs[-1] += 1
                    # And update the total score (+= heatmap score of joint_dst
                    # + score of connecting joint_src with joint_dst)
                    person_limbs[-2] += joint_list[limb_info[1]
                                                       .astype(int), 2] + limb_info[2]
            elif len(person_assoc_idx) == 2:  # if found 2 and disjoint, merge them
                person1_limbs = person_to_joint_assoc[person_assoc_idx[0]]
                person2_limbs = person_to_joint_assoc[person_assoc_idx[1]]
                membership = ((person1_limbs >= 0) & (person2_limbs >= 0))[:-2]
                if not membership.any():  # If both people have no same joints connected, merge them into a single person
                    # Update which joints are connected
                    person1_limbs[:-2] += (person2_limbs[:-2] + 1)
                    # Update the overall score and total count of joints
                    # connected by summing their counters
                    person1_limbs[-2:] += person2_limbs[-2:]
                    # Add the score of the current joint connection to the
                    # overall score
                    person1_limbs[-2] += limb_info[2]
                    person_to_joint_assoc.pop(person_assoc_idx[1])
                else:  # Same case as len(person_assoc_idx)==1 above
                    person1_limbs[joint_dst_type] = limb_info[1]
                    person1_limbs[-1] += 1
                    person1_limbs[-2] += joint_list[limb_info[1]
                                                        .astype(int), 2] + limb_info[2]
            else:  # No person has claimed any of these joints, create a new person
                # Initialize person info to all -1 (no joint associations)
                row = -1 * np.ones(NUM_JOINTS+2)
                # Store the joint info of the new connection
                row[joint_src_type] = limb_info[0]
                row[joint_dst_type] = limb_info[1]
                # Total count of connected joints for this person: 2
                row[-1] = 2
                # Compute overall score: score joint_src + score joint_dst + score connection
                # {joint_src,joint_dst}
                row[-2] = sum(joint_list[limb_info[:2].astype(int), 2]
                              ) + limb_info[2]
                person_to_joint_assoc.append(row)

    # Delete people who have very few parts connected
    people_to_delete = []
    for person_id, person_info in enumerate(person_to_joint_assoc):
        if person_info[-1] < 3 or (person_info[-2] / person_info[-1] < 0.2):
            people_to_delete.append(person_id)

    # Traverse the list in reverse order so we delete indices starting from the
    # last one (otherwise, removing item for example 0 would modify the indices of
    # the remaining people to be deleted!)
    for index in people_to_delete[::-1]:
        person_to_joint_assoc.pop(index)

    # Appending items to a np.array can be very costly (allocating new memory, copying over the array, then adding new row)
    # Instead, we treat the set of people as a list (fast to append items) and
    # only convert to np.array at the end
    return np.array(person_to_joint_assoc)


def plot_pose(img_orig, joint_list, person_to_joint_assoc, bool_fast_plot=True):
    canvas = img_orig.copy()  # Make a copy so we don't modify the original image

    person_num = person_to_joint_assoc.shape[0]
    joints = np.zeros((person_num, 14, 3), dtype=np.float32)
    # print (joints.shape)

    for person, person_joint_info in enumerate(person_to_joint_assoc):
        # print ('-------------find person {}----------------'.format(person))
        for limb_type in range(NUM_LIMBS):
            joint_indices = person_joint_info[joint_to_limb_heatmap_relationship[limb_type]].astype(
                int)
            if -1 in joint_indices or len(joint_indices) < 2:
                # Only draw actual limbs (connected joints), skip if not
                # connected
                continue

            # joint_coords[:,0] represents Y coords of both joints;
            # joint_coords[:,1], X coords
            joint_coords = joint_list[joint_indices, 0:2]
            # print ('joint_indices: {}, joint type: {} , joint_coords: {}'.format(joint_indices, joint_list[joint_indices, -1], joint_coords))

            # find which joint part it is(0=rightShoulder, 1=rightElbow....for ai-challenger format)
            joint_types = joint_list[joint_indices, -1]

            for index, joint_type in enumerate(joint_types):
                x = joint_coords[index][0]
                y = joint_coords[index][1]
                v = 1
                # print (index, joint_type)
                joints[person, int(joint_type), :] = [x, y, v]

            for joint in joint_coords:  # Draw circles at every joint
                cv2.circle(canvas, tuple(joint[0:2].astype(
                    int)), 3, (255, 255, 255), thickness=-1)
                # mean along the axis=0 computes meanYcoord and meanXcoord -> Round
            cv2.line(canvas, tuple(joint_coords[0].astype(int)), tuple(joint_coords[1].astype(int)),
                     color=colors[person % len(colors)], thickness=2)
        # print ('person_joint_info: {}, joints: {}'.format(person_to_joint_assoc, joints))
    # for limb_type in range(NUM_LIMBS):
    #     for person_joint_info in person_to_joint_assoc:
    #         joint_indices = person_joint_info[joint_to_limb_heatmap_relationship[limb_type]].astype(
    #             int)
    #         if -1 in joint_indices or len(joint_indices) < 2:
    #             # Only draw actual limbs (connected joints), skip if not
    #             # connected
    #             continue
    #         # joint_coords[:,0] represents Y coords of both joints;
    #         # joint_coords[:,1], X coords
    #         joint_coords = joint_list[joint_indices, 0:2]
    #
    #         for joint in joint_coords:  # Draw circles at every joint
    #             cv2.circle(canvas, tuple(joint[0:2].astype(
    #                 int)), 3, (255, 255, 255), thickness=-1)
    #             # mean along the axis=0 computes meanYcoord and meanXcoord -> Round
    #         cv2.line(canvas, tuple(joint_coords[0].astype(int)), tuple(joint_coords[1].astype(int)),
    #                  color=colors[limb_type], thickness=2)

    return canvas, joints


def decode_pose(img_orig, param, heatmaps, pafs):

    scale = img_orig.shape[0] / heatmaps.shape[0]

    # Bottom-up approach:
    # Step 1: find all joints in the image (organized by joint type: [0]=nose,
    # [1]=neck...)
    # 这一步就是通过NMS找到所有在heatmap上满足响应阈值的点的坐标, joint_list_per_joint_type是个包含14个list的lists,
    # 14表示总共有14个点的类型. 每个list又是一个包含多个list的lists, 其中每个list是[x, y, score, unique_id, used_flag]类型,
    # x和y表示坐标点位置, score是这个点的分数, unique_id标明这个点在所有点中的id, used_flag用来表示这个点是否被使用过, 用在下面的程序里

    joint_list_per_joint_type = NMS(param, heatmaps, scale)
    # for joint in joint_list_per_joint_type:
    #     print ('....... ', joint)

    # joint_list is an unravel'd version of joint_list_per_joint, where we add
    # a 5th column to indicate the joint_type (0=nose, 1=neck...)

    # joint_list_per_joint_type是个列表的列表中组合, 这行代码是把其变成个ndarray, shape变成(total_joints_num, 6). 其中每一行都是
    # 6个值, 前5个和joint_list_per_joint_type的值一样, 最后一个指代该点的类型, 例如0就代表右肩, 1代表右肘.....

    joint_list = np.array([tuple(peak) + (joint_type,) for joint_type, joint_peaks in enumerate(joint_list_per_joint_type)
                           for peak in joint_peaks])
    # for joint in joint_list:
    #     print ('*....* ', joint)
    # # Step 2: find which joints go together to form limbs (which wrists go
    # # with which elbows)
    paf_upsamp = cv2.resize(pafs, (img_orig.shape[1], img_orig.shape[0]), interpolation=cv2.INTER_CUBIC)

    # 这一步是找到所有可能的连接, connected_limbs是个lists的list, 有多少个需要连接的关节连接个数, 就有多少行, 每一行同样包含不定个数的list,存储的是这个连接所有潜在的连接可能
    # 每一行包含不定的lists, 每个list包含五个值, [joint_src_id,joint_dst_id, limb_score, joint_src_index, joint_dst_index], 前两个用来存储找到的两个点的unique_id,
    # limb_score是计算得到的这个关节连接的分数, joint_src_index和joint_dst_index用来存储找到的点在这个同类型的点的位置. 例如对于头和脖子这个连接,
    # 我们不仅要找到头和脖子这两个点, 并且知道这两个点的unique_id, 还存储了这两个点的在总的type中的位置, 例如这是所有头部点中的第joint_src_index个头部点和所有脖子点中的第joint_dst_index个脖子点
    connected_limbs = find_connected_joints(param, paf_upsamp, joint_list_per_joint_type)

    # Step 3: associate limbs that belong to the same person
    person_to_joint_assoc = group_limbs_of_same_person(connected_limbs, joint_list)

    # for person in person_to_joint_assoc:
    #     print (person.shape)
    #     print (person)


    # (Step 4): plot results
    # joints, ndarray, shape equals to (person_num, 14, 3)
    canvas, joints = plot_pose(img_orig, joint_list, person_to_joint_assoc)

    # for person, joint in enumerate(joints):
    #     assert joint.shape == (14,3)
    #     for i in range(14):
    #         x = int(joint[i][0])
    #         y = int(joint[i][1])
    #         cv2.circle(img_orig, (x, y), 3, (255, 255, 255), thickness=-1)
    #         cv2.putText(img_orig, str(i), (x, y),cv2.FONT_HERSHEY_COMPLEX, 1, colors[person], 1)
    # cv2.imshow('test', img_orig)
    # cv2.waitKey(0)
    joints = np.reshape(joints, (-1, 14*3))
    return canvas, joint_list, person_to_joint_assoc, joints
