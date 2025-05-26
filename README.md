# Football Analysis Project

## Introduction

The goal of this project is to detect and track players, referees, and footballs in a video using YOLO, one of the best AI object detection models available. We will also train the model to improve its performance. Additionally, we will assign players to teams based on the colors of their t-shirts using Kmeans for pixel segmentation and clustering. With this information, we can measure a team's ball acquisition percentage in a match. We will also use optical flow to measure camera movement between frames, enabling us to accurately measure a player's movement. Furthermore, we will implement perspective transformation to represent the scene's depth and perspective, allowing us to measure a player's movement in meters rather than pixels. Finally, we will calculate a player's speed and the distance covered. This project covers various concepts and addresses real-world problems, making it suitable for both beginners and experienced machine learning engineers.

![Screenshot](output_videos/screenshot.png)

## Modules Used

The following modules are used in this project:

- YOLO: AI object detection model
- Kmeans: Pixel segmentation and clustering to detect t-shirt color
- Optical Flow: Measure camera movement
- Perspective Transformation: Represent scene depth and perspective
- Speed and distance calculation per player

## Trained Models

- [Trained Yolo v5](https://drive.google.com/file/d/1DC2kCygbBWUKheQ_9cFziCsYVSRw6axK/view?usp=sharing)

## Sample video

- [Sample input video](https://drive.google.com/file/d/1t6agoqggZKx6thamUuPAIdN_1zR9v9S_/view?usp=sharing)

## Requirements

To run this project, you need to have the following requirements installed:

- Python 3.x
- ultralytics
- supervision
- OpenCV
- NumPy
- Matplotlib
- Pandas

## Usage

### Basic Usage

```bash
python main.py --input input_videos/your_video.mp4
```

### Advanced Options

```bash
# Specify custom output path
python main.py --input input_videos/your_video.mp4 --output output_videos/custom_name.avi

# Process without using stub files (slower but processes everything from scratch)
python main.py --input input_videos/your_video.mp4 --no-stubs

# Force regeneration of stub files even if they exist
python main.py --input input_videos/your_video.mp4 --force-regenerate
```

### Command Line Arguments

- `--input`, `-i`: Path to the input video file (required)
- `--output`, `-o`: Path to save the output video (optional)
- `--no-stubs`: Don't use stub files for faster processing
- `--force-regenerate`: Force regeneration of stub files even if they exist
