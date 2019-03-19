import numpy as np
import imageLoader
import cv2 as cv
import os
from math import floor
from imageLoader import imageResize


##NEXT TO DO: CREATE A SCRIIPT TO POPULATE FOLDERS WITH VARYING DEFAULT THRESHOLLDS AND MHI DURATIONS TO THEN TEST AGAINST ANOTHER VIDEO OR TWO

def populateActionVideos(path):
    limit = 7

    videoPaths = getFilePaths(path, limit)
    print()
    # print("paths: {}".format(videoPaths))

    for newPath in videoPaths:
        addSingleVideoActionFolders(newPath)
        print("Added files from {}".format(path))


def getFilePaths(folder, limit):
    #put all the class folder paths in a list
    fullFilePaths = []
    filesAdded = 0
    for videoName in os.listdir(folder):
        if ".DS_Store" not in videoName and filesAdded < limit:
            fullFilePaths.append(folder + "/" + videoName)
            filesAdded += 1

    return fullFilePaths


def addSingleVideoActionFolders(videoName):
    #MHI length in Frames, evenly spaced out from 20-30 for testing
    for MHI_DURATION in np.linspace(15, 25, 3):
        MHI_DURATION = float("%.4f"%(MHI_DURATION))
        #Threshold is the difference between the greyscale image between frames to record, spaced from 15 to 75 for testing
        for THRESHOLDSHOLD in np.linspace(15, 75, 3):
            THRESHOLDSHOLD = float("%.2f"%(THRESHOLDSHOLD))
            videoToMotionHistoryImage(videoName, MHI_DURATION, THRESHOLDSHOLD)
            # print("Next Video \n ______________________")



def videoToMotionHistoryImage(videoName, MHI_DURATION, THRESHOLD):

    #create the folder the files will save to and return the path
    outputFolder = prepOutputFolder(videoName, MHI_DURATION, THRESHOLD)

    #gives each video file a unique name based on the path to the video
    #currently all manually based on the file directory
    videoFileName = getOutputName(videoName)
    # print("vidfilename: {}".format(videoFileName))

    cap = cv.VideoCapture(videoName, 0)
    currentFrame = 1

    while(cap.isOpened()):

        ret, frame = cap.read()
        if(ret == False):
            break

        h, w = frame.shape[:2]
        prev_frame = frame.copy()
        motion_history = np.zeros((h, w), np.float32)

        #frameRate keeps track of how often to save the image
        frameRate = 5
        #limits the amount of images saved from a video to 7
        totalImagesPerVideo = 7


        while True:

            ret, frame = cap.read()
            if ret == False:
                break

            #finds the absolute difference between the last two frames
            frame_diff = cv.absdiff(frame, prev_frame)
            #makes the image greyscale
            gray_diff = cv.cvtColor(frame_diff, cv.COLOR_BGR2GRAY)
            #records the pixels that are further apart greyscale than the provided threshold
            ret, motion_mask = cv.threshold(gray_diff, THRESHOLD, 1, cv.THRESH_BINARY)
            #keeps track of the motion history in motion_history by using the binary threshold mask above for a length of MHI_DURATION frames
            cv.motempl.updateMotionHistory(motion_mask, motion_history, currentFrame, MHI_DURATION)

            #save the image correct amount of time from motion_history to image to vis
            vis = np.uint8(np.clip((motion_history-(currentFrame-MHI_DURATION)) / MHI_DURATION, 0, 1)*255)
            vis = cv.cvtColor(vis, cv.COLOR_GRAY2BGR)

            #wait until a full MHI Duration is available
            if (currentFrame > MHI_DURATION):
                if((currentFrame + 1) % frameRate == 0 and ((currentFrame / frameRate) < totalImagesPerVideo + 1)):
                    # Used to manually progress every image to be saved
                    # cv.imshow("frame to write", vis)
                    # cv.waitKey(0)

                    #save the image
                    filename =  outputFolder + "/" + videoFileName + "_frame%d.jpg" % currentFrame
                    # print("writing frame {} to file {}".format(currentFrame, filename))
                    cv.imwrite(filename, vis)

            currentFrame += 1
            prev_frame = frame.copy()




    cap.release()
    cv.destroyAllWindows()


def prepOutputFolder(videoName, MHI_DURATION, THRESHOLDSHOLD):
    action = getAction(videoName)

    if "person" in videoName:
        dataset = "KTH"
    else:
        dataset = "Weizman"

    path = dataset + "Actions/MHI_D" + str(MHI_DURATION) + "_thresh" + str(THRESHOLDSHOLD)
    try:
        os.mkdir(path)
    except:
        path = path

    path = path + "/" + action

    try:
        os.mkdir(path)
    except OSError:
        # print("Failed making folder {}".format(path))
        0

    return path

#manually label eaach action based on the label in the name.  Created for Weizman and KHL datasets
def getAction(videoName):
    if "walk" in videoName:
        return "walk"
    elif "wave1" in videoName:
        return "wave1"
    elif "wave2" in videoName:
        return "wave2"
    elif "waving" in videoName:
        return "wave"
    elif "run" in videoName:
        return "run"
    elif "bend" in videoName:
        return "bend"
    elif "jump" in videoName:
        return "jump"
    elif "boxing" in videoName:
        return "boxing"
    elif "jump" in videoName:
        return "jump"
    elif "clap" in videoName:
        return "clap"
    else:
        return "MISCLASSIFIED"

#create the folder the files will save to and return the path
#manually give each output name a unique path
def getOutputName(videoName):
    if "KTH" in videoName:
        if "handwaving" in videoName:
            return videoName[15:]
        elif "boxing" in videoName:
            return videoName[13:]
        elif "handclapping" in videoName:
            return videoName[17:]
        else:
            return videoName[12:]

    else:
        if "run" in videoName:
            return videoName[12:]
        elif "wave" in videoName:
            return videoName[14:]
        else:
            return videoName[13:]



#RUN THESE LINES TO POPULATE THE FOLDERS IN THE DIRECTORY WITH THE SAVED MOTION IMAGE HISTORY
# videoClassFolders = ["Weizman/run", "Weizman/walk", "Weizman/wave1", "Weizman/wave2", "Weizman/jump", "Weizman/bend"]
# for vidClass in videoClassFolders:
#     populateActionVideos(vidClass)
#
# videoClassFolders = ["KTH/running", "KTH/walking", "KTH/handwaving", "KTH/handclapping", "KTH/boxing"]
# for vidClass in videoClassFolders:
#     populateActionVideos(vidClass)
