import os
from pathlib import Path
import tifffile as tiff
import argparse
import xml.etree.ElementTree as ET

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input", type=str, help="Path to the input directory, where multichannel .tiffs are located")
parser.add_argument("-o", "--output", type=str, help="Path to the output directory where the output single-channel .tiffs will be stored")
parser.add_argument("-n", "--channel_names", type=str, default="", help="Comma-separated list of channel names to extract (e.g., 'DAPI,GFP,RFP'). If not provided, all channels will be extracted.")
args = parser.parse_args()

def get_channel_names_from_ome_tiff(tiff_path):
    with tiff.TiffFile(tiff_path) as tif:
        metadata = tif.ome_metadata
    root = ET.fromstring(metadata)
    pixels = root.find('.//{http://www.openmicroscopy.org/Schemas/OME/2016-06}Pixels')
    channel_names = [channel.attrib['Name'] for channel in pixels.findall('.//{http://www.openmicroscopy.org/Schemas/OME/2016-06}Channel')]
    return channel_names

def extract_and_organize_channels_by_name(input_dir, output_dir, channel_names):
    selected_channel_names = [ch.strip() for ch in channel_names.split(',')] if channel_names else []

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    for file_name in os.listdir(input_dir):
        if file_name.endswith('.tiff'):
            file_path = Path(input_dir) / file_name
            base_name = file_name.rsplit('.', 1)[0]
            file_output_path = Path(output_dir) / base_name
            file_output_path.mkdir(parents=True, exist_ok=True)

            try:
                img = tiff.imread(file_path)
                ome_channel_names = get_channel_names_from_ome_tiff(file_path)
                for index, channel_name in enumerate(ome_channel_names):
                    if not selected_channel_names or channel_name in selected_channel_names:
                        channel_img = img[index, :, :]
                        channel_file_name = f"{channel_name}.tiff"
                        channel_file_path = file_output_path / channel_file_name
                        tiff.imwrite(channel_file_path, channel_img)
            except Exception as e:
                print(f"Error processing {file_path}: {e}")

if __name__ == "__main__":
    extract_and_organize_channels_by_name(args.input, args.output, args.channel_names)
