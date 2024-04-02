# Image_analysis_tools

## IMC Analysis

### Preprocessing
#### Extract_single_tiffs.py
A small tool to extract correctly named single-channel .tiff files from Steinbock generated multichannel .tiffs. A Steinbock [Panel_file](https://bodenmillergroup.github.io/steinbock/latest/cli/preprocessing/) needs to be created before.

| Option          | Description                               | Optional |
|-----------------|-------------------------------------------|----------|
| `-i`, `--input`  | Path to the input directory, where multichannel .tiffs are located           | No      |
| `-o`, `--output` | Path to the output directory where the output single-channel .tiffs will be stored      | No      |
| `-p`, `--panel` | Exact path to the panel file   | No       |
| `-c`, `--channels` | "Comma-separated list of channel indices to extract (e.g., '0,1,2')"  | No       |
