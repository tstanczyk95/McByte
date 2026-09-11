import argparse
import os
import os.path as osp
import sys
import time
import cv2
import torch

from loguru import logger

from yolox.data.data_augment import preproc
from yolox.exp import get_exp
from yolox.utils import fuse_model, get_model_info, postprocess
from yolox.utils.visualize import plot_tracking, plot_tracking_basic
from yolox.tracker.mcbyte_tracker import McByteTracker

from mask_propagation.mask_manager import MaskManager

SAM_START_FRAME = 1

OVERLAP_MEASURE_VARIANT_1 = True
OVERLAP_MEASURE_VARIANT_2 = False
GRID_STEP = 10
MASK_CREATION_BB_OVERLAP_THRESHOLD = 0.6

IMAGE_EXT = [".jpg", ".jpeg", ".webp", ".bmp", ".png"]


def make_parser():
    parser = argparse.ArgumentParser("ByteTrack Demo!")
    parser.add_argument(
        "--demo", default="image", help="demo type, possible types: image, video"
    )

    parser.add_argument(
        "--path", help="path to a folder with images (frames) or to a video file"
        # e.g. datasets/SoccerNet/tracking/test/SNMOT-132/img1/
    )
    parser.add_argument(
        "--det_path", 
        default=None, 
        # e.g. 'datasets/SoccerNet/tracking/test/SNMOT-132/det/det.txt'
        help="path to the file with detections; if specified, [detector] related arguments (e.g. exp_file, ckpt) will not be considered"
    )
    parser.add_argument(
        "--save_result",
        default=True,
        action="store_true",
        help="whether to save the inference result of image/video",
    )

    # exp file
    parser.add_argument(
        "-f",
        "--exp_file",
        default="exps/example/mot/yolox_x_mix_det.py",
        # default="exps/example/mot/yolox_x_ablation.py",
        type=str,
        help="[detector] your expriment description file",
    )
    parser.add_argument(
        "-c", "--ckpt",
        default="pretrained/yolox_x_sports_mix.pth.tar", 
        # default="pretrained/bytetrack_ablation.pth.tar", 
        type=str, 
        help="[detector] pretrained model weight (checkpoint) for eval"
    )
    parser.add_argument("-expn", "--experiment-name", type=str, default=None, help="name of the parent output folder (default is the name of the exp_file)")
    parser.add_argument("--experiment_model_name", type=str, default=None, help="[detector] alternative to --exp_file")
    parser.add_argument("--fps", default=30, type=int, help="frame rate (fps)")
    parser.add_argument(
        "--fp16",
        dest="fp16",
        default=True,
        action="store_true",
        help="[detector] Adopting mix precision evaluating.",
    )
    parser.add_argument(
        "--fuse",
        dest="fuse",
        default=True,
        action="store_true",
        help="[detector] Fuse conv and bn for testing.",
    )
    
    # tracking args
    parser.add_argument("--track_thresh", type=float, default=0.6, help="track and detection confidence threshold")
    parser.add_argument("--track_buffer", type=int, default=30, help="number of frames to keep lost tracks")

    # CMC
    parser.add_argument("--cmc-method", default="orb", type=str, help="camera motion compensation method: files (Vidstab GMC) | orb | ecc")

    parser.add_argument("--start_frame_no", type=int, default=1, help="starting frame file number (counting from 1)")
    parser.add_argument("--vis_type", default="basic", type=str, help="visualization type, with OR without detections and tracklets before Kalman filter update OR no visualization: full | basic | no_vis")

    return parser


def get_image_list(path):
    image_names = []
    for maindir, subdir, file_name_list in os.walk(path):
        for filename in file_name_list:
            apath = osp.join(maindir, filename)
            ext = osp.splitext(apath)[1]
            if ext in IMAGE_EXT:
                image_names.append(apath)
    return image_names


def write_results(filename, results):
    save_format = '{frame},{id},{x1},{y1},{w},{h},{s},-1,-1,-1\n'
    with open(filename, 'w') as f:
        for frame_id, tlwhs, track_ids, scores in results:
            for tlwh, track_id, score in zip(tlwhs, track_ids, scores):
                if track_id < 0:
                    continue
                x1, y1, w, h = tlwh
                line = save_format.format(frame=frame_id, id=track_id, x1=round(x1, 1), y1=round(y1, 1), w=round(w, 1), h=round(h, 1), s=round(score, 2))
                f.write(line)
    logger.info('save results to {}'.format(filename))


class Predictor(object):
    def __init__(
        self,
        model,
        exp,
        device=torch.device("cpu"),
        fp16=False
    ):
        self.model = model
        self.num_classes = exp.num_classes
        self.confthre = exp.test_conf
        self.nmsthre = exp.nmsthre
        self.test_size = exp.test_size
        self.device = device
        self.fp16 = fp16
        self.rgb_means = (0.485, 0.456, 0.406)
        self.std = (0.229, 0.224, 0.225)

    def inference(self, img):
        img_info = {"id": 0}
        if isinstance(img, str):
            img_info["file_name"] = osp.basename(img)
            img = cv2.imread(img)
        else:
            img_info["file_name"] = None

        height, width = img.shape[:2]
        img_info["height"] = height
        img_info["width"] = width
        img_info["raw_img"] = img

        img, ratio = preproc(img, self.test_size, self.rgb_means, self.std)
        img_info["ratio"] = ratio
        img = torch.from_numpy(img).unsqueeze(0).float().to(self.device)
        if self.fp16:
            img = img.half()  # to FP16

        with torch.no_grad():
            outputs = self.model(img)
            outputs = postprocess(
                outputs, self.num_classes, self.confthre, self.nmsthre
            )
            #logger.info("Infer time: {:.4f}s".format(time.time() - t0))
        return outputs, img_info
    

def get_detections(img, frame_id, det_list):   
    img_info = {"id": 0}
    if isinstance(img, str):
        img_info["file_name"] = osp.basename(img)
        img = cv2.imread(img)
    else:
        img_info["file_name"] = None

    height, width = img.shape[:2]
    img_info["height"] = height
    img_info["width"] = width
    img_info["raw_img"] = img

    dets_per_frame = [d for d in det_list if d.split(",")[0] == str(frame_id)]

    dets_tensor = torch.zeros(len(dets_per_frame),5) # to be: x1 y1 x2 y2 conf

    for i, line in enumerate(dets_per_frame):
        det = line.split(",") # "frame_id,-1,left,top,width,height,conf,-1,-1,-1"

        dets_tensor[i,0] = int(det[2])
        dets_tensor[i,1] = int(det[3])
        dets_tensor[i,2] = int(det[4]) + int(det[2])
        dets_tensor[i,3] = int(det[5]) + int(det[3])
        dets_tensor[i,4] = int(det[6])

    # To adjust to the format used
    dets_tensor = dets_tensor[None, :]

    dets_array = dets_tensor.numpy()

    return dets_array, img_info


def image_demo(det_source, vis_folder, current_time, args):
    if osp.isdir(args.path):
        files = get_image_list(args.path)
    else:
        files = [args.path]
    files.sort()

    files = files[args.start_frame_no-1:]

    ### For the info logging file save ###
    timestamp = time.strftime("%Y_%m_%d_%H_%M_%S", current_time)
    save_folder = osp.join(vis_folder, timestamp)
    os.makedirs(save_folder, exist_ok=True)
    ### / ###

    vis_type = args.vis_type
    if vis_type not in ['full', 'basic', 'no_vis']:
        print("[vis_type unrecognized, no visualization assumed]")

    if isinstance(det_source, Predictor):
        predictor = det_source
        dets_from_file = False
    elif isinstance(det_source, list):
        det_list = det_source
        dets_from_file = True
    else:
        print("[Unknown type of detection source, exiting.]")
        sys.exit()

    tracker = McByteTracker(args, frame_rate=args.fps, save_folder=save_folder)
    results = []

    # (These 2 lines + required indents) For Cutie
    with torch.inference_mode():
        with torch.cuda.amp.autocast(enabled=True):

            prediction = None
            tracklet_mask_dict = None
            mask_avg_prob_dict = None
            prediction_colors_preserved = None

            mask_menager = MaskManager()

            for frame_id, img_path in enumerate(files, 1):
                print("Frame {}".format(str(frame_id)))
                if dets_from_file:
                    outputs, img_info = get_detections(img_path, frame_id + args.start_frame_no-1, det_list)
                else:
                    outputs, img_info = predictor.inference(img_path)
                
                if outputs[0] is not None:

                    if frame_id > 1:
                        prediction, tracklet_mask_dict, mask_avg_prob_dict, prediction_colors_preserved = mask_menager.get_updated_masks(img_info, img_info_prev, frame_id, online_tlwhs, online_ids, new_tracks, removed_tracks_ids)
                    
                    online_targets, removed_tracks_ids, new_tracks, detections_per_assoc_step, all_considered_tracklets_before_correction = tracker.update(outputs[0], [img_info['height'], img_info['width']], exp.test_size, prediction_mask=prediction, tracklet_mask_dict=tracklet_mask_dict, mask_avg_prob_dict=mask_avg_prob_dict, frame_img=img_info['raw_img'], vis_type=vis_type, dets_from_file=dets_from_file)
                    online_tlwhs = []
                    online_ids = []
                    online_scores = []
                    for t in online_targets:
                        tlwh = t.last_det_tlwh
                        tid = t.track_id
                        online_tlwhs.append(tlwh)
                        online_ids.append(tid)
                        online_scores.append(t.score)
                        # save results
                        results.append(
                            f"{frame_id},{tid},{tlwh[0]:.2f},{tlwh[1]:.2f},{tlwh[2]:.2f},{tlwh[3]:.2f},{t.score:.2f},-1,-1,-1\n"
                        )

                    if vis_type == 'full':
                        considered_online_tlwhs_before_correction = []
                        considered_online_ids_of_tracks_before_correction = []
                        for ct in all_considered_tracklets_before_correction:
                            considered_online_tlwhs_before_correction.append(ct.tlwh)
                            considered_online_ids_of_tracks_before_correction.append(ct.track_id)

                        online_im, online_im_dets, online_im_tracks_before_correction = plot_tracking(
                            img_info['raw_img'], online_tlwhs, online_ids, frame_id=frame_id, prediction_mask=prediction_colors_preserved, det_dict=detections_per_assoc_step, considered_online_tlwhs_before_correction=considered_online_tlwhs_before_correction, considered_online_ids_of_tracks_before_correction=considered_online_ids_of_tracks_before_correction
                        )
                    elif vis_type == 'basic':
                        online_im, online_im_dets, online_im_tracks_before_correction = plot_tracking_basic(
                            img_info['raw_img'], online_tlwhs, online_ids, frame_id=frame_id, prediction_mask=prediction_colors_preserved
                        )
                    else: # 'no_vis'
                        pass
                else:
                    online_im = img_info['raw_img']

                img_info_prev = img_info

                if args.save_result:
                    timestamp = time.strftime("%Y_%m_%d_%H_%M_%S", current_time)
                    save_folder = osp.join(vis_folder, timestamp)
                    os.makedirs(save_folder, exist_ok=True)
                    if vis_type == 'full' or vis_type == 'basic':
                        cv2.imwrite(osp.join(save_folder, osp.basename(img_path)), online_im)
                    if vis_type == 'full':
                        cv2.imwrite(osp.join(save_folder, 'dets__' + osp.basename(img_path)), online_im_dets)
                        if not online_im_tracks_before_correction is None:
                            cv2.imwrite(osp.join(save_folder, 'tr_all__' + osp.basename(img_path)), online_im_tracks_before_correction)

                ch = cv2.waitKey(0)
                if ch == 27 or ch == ord("q") or ch == ord("Q"):
                    break

    if args.save_result:
        res_file = osp.join(vis_folder, f"{timestamp}.txt")
        with open(res_file, 'w') as f:
            f.writelines(results)
        logger.info(f"save results to {res_file}")


def video_demo(det_source, vis_folder, current_time, args):
    ### For the info logging file save ###
    timestamp = time.strftime("%Y_%m_%d_%H_%M_%S", current_time)
    save_folder = osp.join(vis_folder, timestamp)
    os.makedirs(save_folder, exist_ok=True)
    ### / ###

    cap = cv2.VideoCapture(args.path)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)  # float
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)  # float
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Same reason as above: on Windows an absolute --path survives the split,
    # osp.join drops save_folder, and VideoWriter would open the source video
    # for writing. osp.basename is what the image branch already uses.
    save_path = osp.join(save_folder, osp.basename(args.path))
    logger.info(f"video save_path is {save_path}")
    vid_writer = cv2.VideoWriter(
        save_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (int(width), int(height))
    )

    vis_type = args.vis_type
    if vis_type == 'full':
        print("[Visualization of the detections and traclets before KF update is only available for frame-based (image) imput. Doing basic visualization instead]")
        vis_type = 'basic'
    if vis_type not in ['basic', 'no_vis']:
        print("[vis_type unrecognized, no visualization assumed]")

    if isinstance(det_source, Predictor):
        predictor = det_source
        dets_from_file = False
    elif isinstance(det_source, list):
        det_list = det_source
        dets_from_file = True
    else:
        print("[Unknown type of detection source, exiting.]")
        sys.exit()

    tracker = McByteTracker(args, frame_rate=args.fps, save_folder=save_folder)
    results = []

    # (These 2 lines + required indents) For Cutie
    with torch.inference_mode():
        with torch.cuda.amp.autocast(enabled=True):

            prediction = None
            tracklet_mask_dict = None
            mask_avg_prob_dict = None
            prediction_colors_preserved = None

            mask_menager = MaskManager()

            frame_id = 1
            while True:
                ret_val, frame = cap.read()
                if frame_id < args.start_frame_no:
                    frame_id += 1
                    continue
                if not ret_val:
                    break

                print("Frame {}".format(str(frame_id)))
                if dets_from_file:
                    outputs, img_info = get_detections(frame, frame_id, det_list)
                else:
                    outputs, img_info = predictor.inference(frame)
                
                if outputs[0] is not None:

                    if frame_id - args.start_frame_no + 1 > 1:
                        prediction, tracklet_mask_dict, mask_avg_prob_dict, prediction_colors_preserved = mask_menager.get_updated_masks(img_info, img_info_prev, frame_id - args.start_frame_no + 1, online_tlwhs, online_ids, new_tracks, removed_tracks_ids)
                    
                    online_targets, removed_tracks_ids, new_tracks, _, _ = tracker.update(outputs[0], [img_info['height'], img_info['width']], exp.test_size, prediction_mask=prediction, tracklet_mask_dict=tracklet_mask_dict, mask_avg_prob_dict=mask_avg_prob_dict, frame_img=img_info['raw_img'], vis_type=vis_type, dets_from_file=dets_from_file)
                    online_tlwhs = []
                    online_ids = []
                    online_scores = []
                    for t in online_targets:
                        tlwh = t.last_det_tlwh
                        tid = t.track_id
                        online_tlwhs.append(tlwh)
                        online_ids.append(tid)
                        online_scores.append(t.score)
                        # save results
                        results.append(
                            f"{frame_id},{tid},{tlwh[0]:.2f},{tlwh[1]:.2f},{tlwh[2]:.2f},{tlwh[3]:.2f},{t.score:.2f},-1,-1,-1\n"
                        )

                    if vis_type == 'basic':
                        online_im, _, _ = plot_tracking_basic(
                            img_info['raw_img'], online_tlwhs, online_ids, frame_id=frame_id, prediction_mask=prediction_colors_preserved
                        )
                    else: # 'no_vis'
                        pass
                else:
                    online_im = img_info['raw_img']

                img_info_prev = img_info

                if args.save_result:
                    timestamp = time.strftime("%Y_%m_%d_%H_%M_%S", current_time)
                    save_folder = osp.join(vis_folder, timestamp)
                    os.makedirs(save_folder, exist_ok=True)
                    vid_writer.write(online_im)                   

                ch = cv2.waitKey(1)
                if ch == 27 or ch == ord("q") or ch == ord("Q"):
                    break

                frame_id += 1

    vid_writer.release()        

    if args.save_result:
        res_file = osp.join(vis_folder, f"{timestamp}.txt")
        with open(res_file, 'w') as f:
            f.writelines(results)
        logger.info(f"save results to {res_file}")


def main(exp, args):
    if not args.experiment_name:
        args.experiment_name = exp.exp_name

    output_dir = osp.join(exp.output_dir, args.experiment_name)
    os.makedirs(output_dir, exist_ok=True)

    if args.save_result:
        vis_folder = osp.join(output_dir, "track_vis")
        os.makedirs(vis_folder, exist_ok=True)
    device = "cuda:0"

    logger.info("Args: {}".format(args))

    det_path = args.det_path
    if det_path is not None:
        # load detections from a file
        with open(det_path, "r") as f_det:
            det_list = f_det.readlines()
        logger.info("Using detections from the file: {}".format(det_path))
        det_source = det_list
    else:
        # activate detector
        model = exp.get_model().to(device)
        logger.info("Model Summary: {}".format(get_model_info(model, exp.test_size)))
        model.eval()

        if args.ckpt is None:
            ckpt_file = osp.join(output_dir, "best_ckpt.pth.tar")
        else:
            ckpt_file = args.ckpt
        logger.info("loading checkpoint")
        ckpt = torch.load(ckpt_file, map_location="cpu")
        # load the model state dict
        model.load_state_dict(ckpt["model"])
        logger.info("loaded checkpoint done.")

        if args.fuse:
            logger.info("\tFusing model...")
            model = fuse_model(model)

        if args.fp16:
            model = model.half()  # to FP16

        predictor = Predictor(model, exp, device, args.fp16)
        det_source = predictor

    current_time = time.localtime()
    if args.demo == "image":
        image_demo(det_source, vis_folder, current_time, args)
    elif args.demo == "video":
        video_demo(det_source, vis_folder, current_time, args)
    else:
        print("[No valid input mode selected (--demo=...). Tracking not performed. Available modes: image, video]")


if __name__ == "__main__":
    args = make_parser().parse_args()
    exp = get_exp(args.exp_file, args.experiment_model_name)

    main(exp, args)
