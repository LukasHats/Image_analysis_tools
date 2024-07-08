import os
from pathlib import Path
import tifffile as tiff
import argparse
import csv

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input", type=str, help="Path to the input directory, where multichannel .tiffs are located")
parser.add_argument("-o", "--output", type=str, help="Path to the output directory where the output single-channel .tiffs will be stored")
parser.add_argument("-p", "--panel", type=str, help="Path to the panel file")
parser.add_argument("-c", "--channels", type=str, default=None, help="Comma-separated list of channel indices to extract (e.g., '0,1,2'). If not provided, all channels will be extracted.")
args = parser.parse_args()

def read_panel_file(panel_path):
    channel_info = []
    with open(panel_path, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            channel_name = f"{row['channel']}_{row['name']}".strip("_")
            channel_info.append(channel_name)
    return channel_info

def extract_and_organize_channels(input_dir, output_dir, panel_path, channels):
    channel_info = read_panel_file(panel_path)

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    for file_name in os.listdir(input_dir):
        if file_name.endswith('.tiff'):
            file_path = Path(input_dir) / file_name
            base_name = file_name.rsplit('.', 1)[0]
            file_output_path = Path(output_dir) / base_name
            file_output_path.mkdir(parents=True, exist_ok=True)

            try:
                img = tiff.imread(file_path)
                if channels:
                    selected_channels = [int(ch.strip()) for ch in channels.split(',')]
                else:
                    # If channels are not specified, extract all channels
                    selected_channels = range(img.shape[0])

                for channel in selected_channels:
                    channel_img = img[channel, :, :]
                    channel_label = channel_info[channel] if channel < len(channel_info) else f"Channel{channel+1}"
                    channel_file_name = f"{channel_label}.tiff"
                    channel_file_path = file_output_path / channel_file_name
                    tiff.imwrite(channel_file_path, channel_img)
            except Exception as e:
                print(f"Error processing {file_path}: {e}")

if __name__ == "__main__":
    extract_and_organize_channels(args.input, args.output, args.panel, args.channels)
