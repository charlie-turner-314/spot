import os
import json

def spot_to_csv(spot_dict, spot_csv):

    with open(spot_csv, "w") as spot_csv:
        spot_csv.write("video,label,frame,confidence\n")
        for vid in spot_dict:
            video = vid["video"]
            for event in vid["events"]:
                spot_csv.write(f"{video},{event['label']},{event['frame']},{event['score']}\n")

