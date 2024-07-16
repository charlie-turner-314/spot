import os
import re
import json
import torch
import numpy as np

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


def infer_video(
    video_path,
) -> tuple:
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
    video_name = splits["video_name"]

    # need to have a 'labels' file. We can create a dummy one
    labels = [
        {
            "video": video_name,
            "events": [],
            "fps": 60,
            "num_frames": splits["num_frames"],
        }
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

    return result[0], splits["num_frames"]


def evaluate(model, dataset, classes) -> "list[dict]":
    pred_dict = {}
    for video, video_len, _ in dataset.videos:
        pred_dict[video] = (
            np.zeros((video_len, len(classes) + 1), np.float32),
            np.zeros(video_len, np.int32),
        )

    # Do not up the batch size if the dataset augments
    batch_size = 16

    for clip in tqdm(
        DataLoader(
            dataset,
            num_workers=8,
            pin_memory=True,
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


def split_video(video_path, frame_dir) -> dict:
    """Split video into frames with ffmpeg

    Returns:
        dict: {frames_path: str, frame_rate: int, num_frames: int}
    """

    video_name = os.path.basename(video_path).split(".")[0]

    if os.path.exists(os.path.join(frame_dir, video_name)):
        # remove existing frames
        os.system(f"rm -r {os.path.join(frame_dir, video_name)}")

    os.makedirs(os.path.join(frame_dir, video_name))

    os.system(
        f"ffmpeg -i {video_path} -vf 'scale=-1:256' -qscale:v 2 {frame_dir}/{video_name}/%06d.jpg -loglevel quiet"
    )

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
    results = infer_video(vid)
    # prettyprint ndarray as json
    print(results)
