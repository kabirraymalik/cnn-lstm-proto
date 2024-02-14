import torch
import main 
import os
import cv2

def generateFrames(numFrames):
    path = os.getcwd()
    if os.path.exists(path + "data/image_data/") == False:
        os.mkdir(path+"data/image_data/")
    videoData = os.fsencode(path+"data/video_data/")
    for label in os.listdir(videoData):
        os.mkdir(path+"data/image_data/"+label+"/")
        for video in os.listdir(videoData+label+"/"):
            os.mkdir(path+"data/image_data/" + label + "/" + video)
            #TODO: extract frames from video with cv2

