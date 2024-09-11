import json
import json
import os
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip


# Function to identify point sequences
def identify_point_sequences(data, time_threshold=60, confidence_threshold=0.8):
    """
    Identify sequences of points in a CSV file based on time and confidence thresholds.
    Parameters:
    - csv (str): The path to the CSV file containing the data.
    - time_threshold (int, optional): The maximum time difference (in frames) between consecutive points in a sequence. Defaults to 60.
    - confidence_threshold (float, optional): The minimum confidence level required for a point to be included in a sequence. Defaults to 0.8.
    Returns:
    - sequences (list): A list of sequences, where each sequence is a list of rows from the CSV file.
    """
    # Filter based on confidence threshold
    data_filtered = data[data["confidence"] >= confidence_threshold]
    # Sort by frame
    data_filtered = data_filtered.sort_values(by="frame").reset_index(drop=True)

    sequences = []
    current_sequence = []
    previous_frame = None

    for idx, row in data_filtered.iterrows():
        # Check if the current event is close in time to the previous event
        if previous_frame is None or row["frame"] - previous_frame <= time_threshold:
            current_sequence.append(row)
        else:
            if len(current_sequence) > 0:
                sequences.append(current_sequence)
            current_sequence = [row]
        previous_frame = row["frame"]

    if len(current_sequence) > 0:
        sequences.append(current_sequence)

    return sequences

def filter_and_condense_sequences(sequences, merge_threshold=5):
    """
    Filter and condense sequences by merging similar events close in time.
    Parameters:
    - sequences (list): A list of sequences, where each sequence is a list of events.
    - merge_threshold (int, optional): The time threshold for merging similar events. Default is 5.
    Returns:
    - condensed_sequences (list): A list of condensed sequences, where each condensed sequence is a list of events.
    """

    condensed_sequences = []

    for sequence in sequences:
        condensed_sequence = []
        last_event = None

        for idx, event in enumerate(sequence):
            if last_event is None:
                condensed_sequence.append(event)
                last_event = event
            else:
                # Merge similar events close in time
                if (
                    event["label"] == last_event["label"]
                    and (event["frame"] - last_event["frame"]) <= merge_threshold
                ):
                    continue
                else:
                    condensed_sequence.append(event)
                    last_event = event

        if len(condensed_sequence) > 1:  # Only keep sequences with more than one event
            condensed_sequences.append(condensed_sequence)

    return condensed_sequences




def extract_sequences(sequences, video_path, output_dir, sequence_dir="seqs", frame_rate=25):
    new_sequences = []
    for i, sequence in enumerate(sequences):
        # if "kyr" not in fg_player[i]:  # only keep kyrgios sequences
        #     continue

        start_frame = sequence[0]["frame"]
        end_frame = sequence[-1]["frame"]

        # check if the sequence is long enough ~2 seconds
        if end_frame - start_frame < (frame_rate * 2):
            continue

        new_sequences.append(
            [
                {
                    "frame": start_frame,
                    # "label": fg_player[i],
                    "confidence": 1,
                },
                {
                    "frame": end_frame,
                    # "label": fg_player[i],
                    "confidence": 1,
                },
            ]
        )


    os.makedirs(output_dir, exist_ok=True)
    video_name = os.path.basename(video_path).split(".")[0]
    # Process each sequence
    for i, sequence in enumerate(new_sequences):
        start_frame = sequence[0]["frame"] - 10
        end_frame = sequence[-1]["frame"] + 10

        # Convert frames to seconds
        start_time = start_frame / frame_rate
        end_time = end_frame / frame_rate

        # Generate output file name 3 digit index
        output_file = os.path.join(output_dir, f"{video_name}_{i:03d}.mp4")

        # Extract the subclip from the video
        ffmpeg_extract_subclip(video_path, start_time, end_time, targetname=output_file)

        # save the events in the sequence to a json file in aux/
        sequence_file = os.path.join(sequence_dir, os.basename(output_file).split(".")[0] + ".json")
        # account for the start frame offset
        first_event = sequence[0] 
        for event in sequence:
            event["frame"] -= (first_event["frame"] - 10)

        with open(sequence_file, "w") as f:
            json.dump(sequence, f)







    print("Extraction complete.")
