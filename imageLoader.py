import cv2 as cv
import numpy as np    # for mathematical operations
import os
import l1pca
import random
import math
import fnmatch


#IN: a string of the folder the faces are in, and the train/testsplit percentage you want to have
#OUT: returns the full list of the test data,

def loadFaces(facesFolder, trainSplitPercentage = 0.7, pcaInitializations = 10, kPrincipalComponents = 6):
    testData = []
    QpropList = []
    classMeanList = []

    classIterations = 1

    maximumClassLimit, maximumPhotoLimit, badImages = folderSetUp(facesFolder)

    fullFilePaths = getFolderPaths(facesFolder)

    for fullFilePath in fullFilePaths:
        # print("Current file path: {}".format(fullFilePath))
        compiledImages = []
        if classIterations > maximumClassLimit:
            break

        if(os.path.isdir(fullFilePath) and ".DS_Store" not in fullFilePath):
            photoCount = 0

            for filename in os.listdir(fullFilePath):
                if photoCount > maximumPhotoLimit:
                    break

                #Make sure that the image was not one of the really dark ones
                if checkImageQuality(filename, badImages):
                    #the zero flag reads the image in grayscale to avoid redundancy
                    currentImage = cv.imread(fullFilePath + filename, 0)
                    # cv.imshow('Original Imgae', currentImage)
                    # cv.waitKey(0)
                    # print("Current Image shape before resize: {}".format(currentImage.shape))
                    currentImage = imageResize(currentImage, width = 60)
                    # cv.imshow('Resized Image', currentImage)
                    # cv.waitKey(0)
                    # print("Current Image shape after resize: {}".format(currentImage.shape))

                    flattenedImage = currentImage.flatten()

                    compiledImages.append(flattenedImage)
                    photoCount += 1


            random.shuffle(compiledImages)

            splitNumber = math.floor(len(compiledImages) * trainSplitPercentage)
            trainData = np.array(compiledImages[0:splitNumber])


            if not testData:
                testData = compiledImages[splitNumber:] #+ ["NEXT"]
                # print("Adding {} photos to training from the path {}".format((len(compiledImages)-splitNumber), fullFilePath))
            else:
                # print("Adding {} photos to training from the path {}".format((len(compiledImages)-splitNumber), fullFilePath))
                testData = testData + compiledImages[splitNumber:]

            classIterations +=1


            ######
            #run l1pca on trainData here
            ######

            trainData, classMean = prepPca(trainData)
            classMeanList.append(classMean)

            kComponentQprop = []
            #run PCA for each principal component, removing the principal component for each iteration in training data
            for principalComponentIterations in range(0, kPrincipalComponents):
                Qprop, updatedTrainData = runPca(trainData, pcaInitializations)

                #drop the dimension of qprop by 1 so it just appends the principal component
                kComponentQprop.append(np.squeeze(Qprop))
                trainData = updatedTrainData


            QpropList.append(kComponentQprop)


    QpropList = np.array(QpropList)
    testData = np.array(testData)
    classMeanList = np.array(classMeanList)
    cv.destroyAllWindows()
    # print("FINAL SHAPES: \n  QpropList: {} \n  testData: {}\n  classMeanList: {}".format(
    #     QpropList.shape, testData.shape, classMeanList.shape))
    #
    # print("final Qprop for class 0 {}".format(QpropList[0]))

    return testData, QpropList, classMeanList


def prepPca(trainData):
    #Zeros out the feature means, then it transposes the data

    classMean = np.mean(trainData, axis = 0)
    trainData = trainData - classMean
    #print("class mean shape {}\n train data shape {}".format(classMean.shape, trainData.shape))
    # mean shape is 10304, data shape is 7, 10304

    trainData = np.transpose(trainData)

    return trainData, classMean



#https://stackoverflow.com/questions/44650888/resize-an-image-without-distortion-opencv
def imageResize(image, width = None, height = None, inter = cv.INTER_AREA):
    # initialize the dimensions of the image to be resized and
    dim = None
    (h, w) = image.shape[:2]

    if width is None and height is None:
        return image

    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the dimensions
        r = height / float(h)
        dim = (int(w * r), height)

    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    resized = cv.resize(image, dim, interpolation = inter)

    return resized


def folderSetUp(facesFolder):
    #just set up to run on orl_faces and CroppedYale right now
    if facesFolder == "orl_faces":
        maximumPhotoLimit = 10
        maximumClassLimit = 8
        badImages =[]

    else:
        maximumPhotoLimit = 25
        maximumClassLimit = 8
        badImages = ["Ambient", "_P00A+130E+20", "_P00A+120E+00", "_P00A+110E+65", "_P00A+110E+40", "_P00A+110E+15", "_P00A+110E-20",
        "_P00A+095E+00", "_P00A+085E+20", "_P00A+085E-20", "_P00A+070E+45", "_P00A+070E+00", "_P00A+070E-35",
        "_P00A+050E-40", "_P00A+000E+90", "P00A-130E+20", "_P00A-120E+00", "_P00A-110E+65", "_P00A-110E+40", "_P00A-110E+15"
        "_P00A-110E-20", "_P00A-095E+00", "_P00A-085E+20", "_P00A-085E-20", "_P00A-070E+45", "_P00A-035E+65"]
        # removed "_P00A-070E+00", "_P00A-070E-35",

    return maximumClassLimit, maximumPhotoLimit, badImages


def checkImageQuality(filename, badImages):
    if not filename.endswith(".pgm"):
        # print("File extension {} incorrect ".format(filename))
        return False

    for badImageLabel in badImages:
        if badImageLabel in filename:
            # print("File name {} incorrect".format(filename))
            return False

    return True



def runPca(trainData, pcaInitializations):
    Qprop = l1pca.l1pca(trainData, pcaInitializations)

    Qprop = np.expand_dims(Qprop, axis = 0)
    Qmult = np.matmul(Qprop.transpose(), Qprop)

    # print("Qmult shape: {}, trainData shape : {}".format(Qmult.shape, trainData.shape))
    updatedTrainData = trainData - np.matmul(Qmult, trainData)
    # print("updatedTrainData shape: {}, trainData shape : {}".format(updatedTrainData.shape, trainData.shape))

    return Qprop, updatedTrainData


def getFolderPaths(facesFolder):
    #put all the class folder paths in a list
    fullFilePaths = []
    for faceClassFolder in os.listdir(facesFolder):
        fullFilePaths.append(facesFolder + "/" + faceClassFolder + "/")
    #Randomize the order of the image paths
    random.shuffle(fullFilePaths)

    return fullFilePaths

#
#
# facesFolder = "orl_faces"
# testData, Qprop, classMeanList = loadFaces(facesFolder)
# print("Test Data: {}".format(testData))
