import torch
import os
import ffmpeg
import matplotlib.pyplot as plt
import numpy as np
 
def generateFrames(numFrames):
    path = os.getcwd()
    if os.path.exists(path + "/data/image_data/") == False:
        os.mkdir(path+"/data/image_data/")
    else:
        print("/data/image_data/ already exists!")
        return 0
    videoData = os.fsencode(path+"/data/video_data/")
    for labelobj in os.listdir(videoData):
        label = str(labelobj)[2:-1]
        os.mkdir(path+"/data/image_data/"+label+"/")
        for video in os.listdir(path+"/data/video_data/"+label):
            dirName = path+"/data/image_data/" + label + "/" + video[0:-4]
            os.mkdir(dirName)
            probe = ffmpeg.probe(path+"/data/video_data/"+label+"/"+video)
            video_info = next(stream for stream in probe['streams'] if stream['codec_type'] == 'video')
            duration = float(video_info['duration'])
            # Calculate intervals for frame extraction
            intervals = np.linspace(0, duration, numFrames, endpoint=False)
            for i, interval in enumerate(intervals):
                output_file = f"frame_{i+1}.jpg"  # Define your output frame filename
                # Extract frame
                (
                ffmpeg
                .input(path+"/data/video_data/"+label+"/"+video, ss=interval)  # Seek to position
                .output(path+"/data/image_data/"+label+"/"+video[0:-4]+"/"+output_file, vframes=1)  # Extract one frame
                .run(capture_stdout=True, capture_stderr=True)
                )
        progress = os.listdir(videoData).index(labelobj)/len(os.listdir(videoData)) * 100
        print("building... " + str(round(progress,4)) + "% \complete")
    return 1

