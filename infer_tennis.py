import os
import re
import json
import torch
import numpy as np

import sys
# allow the following imports
sys.path.append("/home/charlie/Documents/ATPIL/dataset_preparation/spot")

from tqdm import tqdm
from .dataset.frame import ActionSpotVideoDataset
from .util.io import load_json
from .util.eval import process_frame_predictions
from .util.dataset import load_classes
from .train_e2e import E2EModel
from torch.utils.data import DataLoader


config = {
    "batch_size": 8,
    "clip_len": 100,
    "crop_dim": 224,
    "dataset": "tennis",
    "dilate_len": 0,
    "epoch_num_frames": 500000,
    "feature_arch": "rny008_gsm",
    "fg_upsample": None,
    "gpu_parallel": False,
    "learning_rate": 0.001,
    "mixup": True,
    "modality": "rgb",
    "num_classes": 6,
    "num_epochs": 50,
    "start_val_epoch": 30,
    "temporal_arch": "gru",
    "warm_up_epochs": 3,
}


def get_last_epoch(model_dir):
    regex = re.compile(r"checkpoint_(\d+)\.pt")

    last_epoch = -1
    for file_name in os.listdir(model_dir):
        m = regex.match(file_name)
        if m:
            epoch = int(m.group(1))
            last_epoch = max(last_epoch, epoch)
    assert last_epoch >= 0
    return last_epoch

def clean_frame_dir(frame_dir):
    """Remove all frames in the frame directory"""
    os.system(f"rm -r {frame_dir}/*")

def infer_frame_dir(frame_dir):
    model_dir = "/home/charlie/Documents/ATPIL/dataset_preparation/spot/model_dir"
    dataset = "tennis"
    classes = load_classes(
        "/home/charlie/Documents/ATPIL/dataset_preparation/spot/data/tennis/class.txt"
    )
    num_classes = len(classes) + 1  # add one for the 'null' class?
    best_epoch = get_last_epoch(model_dir)
    model = E2EModel(
        num_classes,
        config["feature_arch"],
        config["temporal_arch"],
        clip_len=config["clip_len"],
        modality=config["modality"],
        multi_gpu=config["gpu_parallel"],
    )
    model.load(
        torch.load(os.path.join(model_dir, "checkpoint_{:03d}.pt".format(best_epoch)))
    )
    # need to have a 'labels' file. We can create a dummy one
    labels = [
        {
            "video": video_name,
            "events": [],
            "fps": 25,
            "num_frames": len(os.listdir(os.path.join(frame_dir, video_name))), 
            }
        for video_name in os.listdir(frame_dir)
    ]

    if os.path.exists("labels.json"):
        os.remove("labels.json")

    with open("labels.json", "w") as f:
        json.dump(labels, f)

    # Dataset
    dataset = ActionSpotVideoDataset(
        classes,
        "labels.json",
        frame_dir,
        config["modality"],
        config["clip_len"],
        overlap_len=config["clip_len"] // 2,
        crop_dim=config["crop_dim"],
    )

    result = evaluate(model, dataset, classes)

    return result

def infer_video(
    video_path,
) -> dict:
    """
    Use the model to infer keyframes from videos
    """
    model_dir = "/home/charlie/Documents/ATPIL/dataset_preparation/spot/model_dir"
    dataset = "tennis"
    classes = load_classes(
        "/home/charlie/Documents/ATPIL/dataset_preparation/spot/data/tennis/class.txt"
    )
    num_classes = len(classes) + 1  # add one for the 'null' class?

    # Model
    best_epoch = get_last_epoch(model_dir)
    model = E2EModel(
        num_classes,
        config["feature_arch"],
        config["temporal_arch"],
        clip_len=config["clip_len"],
        modality=config["modality"],
        multi_gpu=config["gpu_parallel"],
    )
    model.load(
        torch.load(os.path.join(model_dir, "checkpoint_{:03d}.pt".format(best_epoch)))
    )

    frame_dir = "frame_dir"

    splits = split_video(video_path, frame_dir)
    print("Num frames:", splits["num_frames"])
    video_name = splits["video_name"]

    # need to have a 'labels' file. We can create a dummy one
    labels = [
        {
            "video": video_name,
            "events": [],
            "fps": 25,
            "num_frames": splits["num_frames"],
        } ]
    if os.path.exists("labels.json"):
        os.remove("labels.json")

    with open("labels.json", "w") as f:
        json.dump(labels, f)

    # Dataset
    dataset = ActionSpotVideoDataset(
        classes,
        "labels.json",
        frame_dir,
        config["modality"],
        config["clip_len"],
        overlap_len=config["clip_len"] // 2,
        crop_dim=config["crop_dim"],
    )

    result = evaluate(model, dataset, classes)

    events = result[0]

    return filter_events(events)

def filter_events(events, threshold=5):
    # remove events that are too close together
    last_event = None
    filtered_events = []
    for event in events["events"]:
        if last_event is None:
            last_event = event
            filtered_events.append(event)
        else:
            if event["frame"] - last_event["frame"] < threshold:
                # keep only the one with the highest confidence score
                if event["score"] > last_event["score"]:
                    filtered_events[-1] = event
            else:
                filtered_events.append(event)
                last_event = event
    events["events"] = filtered_events
    return events



def evaluate(model, dataset, classes) -> "list[dict]":
    pred_dict = {}
    for video, video_len, _ in dataset.videos:
        pred_dict[video] = (
            np.zeros((video_len, len(classes) + 1), np.float32),
            np.zeros(video_len, np.int32),
        )

    # Do not up the batch size if the dataset augments
    batch_size = 1

    for clip in tqdm(
        DataLoader(
            dataset,
            num_workers=8,
            # pin_memory=True,
            batch_size=batch_size,
        )
    ):
        if batch_size > 1:
            # Batched by dataloader
            _, batch_pred_scores = model.predict(clip["frame"])

            for i in range(clip["frame"].shape[0]):
                video = clip["video"][i]
                scores, support = pred_dict[video]
                pred_scores = batch_pred_scores[i]
                start = clip["start"][i].item()
                if start < 0:
                    pred_scores = pred_scores[-start:, :]
                    start = 0
                end = start + pred_scores.shape[0]
                if end >= scores.shape[0]:
                    end = scores.shape[0]
                    pred_scores = pred_scores[: end - start, :]
                scores[start:end, :] += pred_scores
                support[start:end] += 1

        else:
            # Batched by dataset
            scores, support = pred_dict[clip["video"][0]]

            start = clip["start"][0].item()
            _, pred_scores = model.predict(clip["frame"][0])
            if start < 0:
                pred_scores = pred_scores[:, -start:, :]
                start = 0
            end = start + pred_scores.shape[1]
            if end >= scores.shape[0]:
                end = scores.shape[0]
                pred_scores = pred_scores[:, : end - start, :]

            scores[start:end, :] += np.sum(pred_scores, axis=0)
            support[start:end] += pred_scores.shape[0]

    err, f1, pred_events, pred_events_high_recall, pred_scores = (
        process_frame_predictions(dataset, classes, pred_dict)
    )

    # Return predictions
    return pred_events

import os
import subprocess
from tqdm import tqdm

def extract_frames_with_progress(video_path, frame_dir, video_name):
    # Get the total number of frames in the video with ffprobe
    result = subprocess.run(
        [
            "ffprobe", 
            "-v", "error", 
            "-select_streams", "v:0", 
            "-count_packets", 
            "-show_entries", "stream=nb_read_packets", 
            "-of", "csv=p=0", 
            video_path
        ],
        stdout=subprocess.PIPE, 
        stderr=subprocess.PIPE, 
        text=True
    )

    total_frames = int(result.stdout.strip())


    print(f"Extracting {total_frames} frames from the video")
    
    # Use subprocess to run ffmpeg and capture the output
    process = subprocess.Popen(
        [
            "ffmpeg", 
            "-i", video_path, 
            "-vf", "scale=-1:256", 
            "-qscale:v", "2", 
            os.path.join(frame_dir, video_name, "%06d.jpg")
        ],
        stdout=subprocess.PIPE, 
        stderr=subprocess.PIPE, 
        text=True
    )
    
    # Create a progress bar with tqdm
    with tqdm(total=total_frames, desc="Extracting frames", unit="frame") as pbar:
        while True:
            output = process.stderr.read(100)
            if output == "" and process.poll() is not None:
                break
            if "frame=" in output:
                # Extract the current frame number from the ffmpeg output
                try:
                    current_frame = int(output.split("frame=")[-1].split()[0])
                    pbar.update(current_frame - pbar.n)
                except:
                    pass
    
    # Wait for the subprocess to finish
    res = process.wait()
    if res != 0:
        print("Error extracting frames")
        exit()



def split_video(video_path, frame_dir) -> dict:
    """Split video into frames with ffmpeg

    Returns:
        dict: {frames_path: str, frame_rate: int, num_frames: int}
    """

    video_name = os.path.basename(video_path).split(".")[0]

    if os.path.exists(os.path.join(frame_dir, video_name)):
        print("frames exist, deleting")
        # remove existing frames
        os.system(f"rm -rf {os.path.join(frame_dir, video_name)}")

    os.makedirs(os.path.join(frame_dir, video_name))

    os.system(
        f"ffmpeg -i {video_path} -vf 'scale=-1:256' -qscale:v 2 {frame_dir}/{video_name}/%06d.jpg -loglevel quiet"
    )
    # extract_frames_with_progress(video_path, frame_dir, video_name)

    num_frames = len(os.listdir(os.path.join(frame_dir, video_name)))

    result = {
        "video_name": video_name,
        "frame_rate": None,  # TODO: get frame rate
        "num_frames": num_frames,
    }
    return result


if __name__ == "__main__":
    print("This script is not meant to be run directly")
    vid = input("Enter video path: ")
    results, num_frames = infer_video(vid)
    # save to file
    with open("spotresults.json", "w") as f:
        json.dump(results, f)
    print(f"Results saved to 'results.json' with {num_frames} frames")
