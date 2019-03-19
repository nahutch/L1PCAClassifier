#VIDEO LOADER
import cv2 as cv
import numpy as np    # for mathematical operations
import os
import l1pca
import random
import math


def loadActions(actionsFolder, trainSplitPercentage = 0.7, pcaInitializations = 10, kPrincipalComponents = 6, testSet = 0):
    #Variables to return.  testData holds the data to test.  All three variables are correlated at each index of the list.
    #Ex: the third test data value, testData[3]'s Qprop value is at QpropList[3] and the class mean created from its test data at classMeanList[3]
    testData = []
    QpropList = []
    classMeanList = []

    classIterations = 1

    #based on which folder is running, manually define a maximumClass and maxinumPhoto limit
    maximumClassLimit, maximumPhotoLimit = folderSetUp(actionsFolder)

    #retrieve the photo paths of the photos
    fullFilePaths = getFolderPaths(actionsFolder, maximumClassLimit, testSet)

    for fullFilePath in fullFilePaths:
        if classIterations > maximumClassLimit:
            break

        #print("Current file path: {}".format(fullFilePath))
        #currentImages keeps track of every photo from the class folder
        compiledImages = []

        if(os.path.isdir(fullFilePath) and ".DS_Store" not in fullFilePath):
            photoCount = 0

            for filename in os.listdir(fullFilePath):
                if photoCount > maximumPhotoLimit:
                    break

                #Make sure that the image exists, is not blacklisted, or is not just fully black
                if checkImageQuality(fullFilePath, filename):
                    #the zero flag reads the image in grayscale to avoid redundancy
                    currentImage = cv.imread(fullFilePath + filename, 0)
                    # cv.imshow('Original Imgae', currentImage)
                    # cv.waitKey(0)
                    currentImage = imageResize(currentImage, width = 60)

                    #flatten the image
                    flattenedImage = currentImage.flatten()

                    #keep track of the flattened image as a vector in the compiledImages list
                    compiledImages.append(flattenedImage)
                    photoCount += 1


            #shuffle the data to make sure different train/test sets each iteration
            random.shuffle(compiledImages)

            #split the train and test data
            splitNumber = math.floor(len(compiledImages) * trainSplitPercentage)
            trainData = np.array(compiledImages[0:splitNumber])


            if not testData:
                testData = compiledImages[splitNumber:]
                # print("Adding {} photos to training from the path {}".format((len(compiledImages)-splitNumber), fullFilePath))
            else:
                # print("Adding {} photos to training from the path {}".format((len(compiledImages)-splitNumber), fullFilePath))
                testData = testData + compiledImages[splitNumber:]

            classIterations +=1

            if photoCount < maximumPhotoLimit:
                print("Not enough photos, exiting")
                exit(1)

            ######
            #run l1pca on trainData here
            ######

            #prepPCA zeros out the mean and formats it so it can enter PCA
            trainData, classMean = prepPca(trainData)
            #rercord the class mean for testing purposes later
            classMeanList.append(classMean)

            #keep tracks of each K component through the PCA iterations
            kComponentQprop = []
            #run PCA for each principal component, removing the principal component for each iteration in training data to save
            for principalComponentIterations in range(0, kPrincipalComponents):

                Qprop, updatedTrainData = runPca(trainData, pcaInitializations)

                #drop the dimension of qprop by 1 so it just appends the principal component, not a vector of the principal component
                kComponentQprop.append(np.squeeze(Qprop))
                trainData = updatedTrainData


            QpropList.append(kComponentQprop)

    #change all arrays to be np arrays and close windows if any existed
    QpropList = np.array(QpropList)
    testData = np.array(testData)
    classMeanList = np.array(classMeanList)
    cv.destroyAllWindows()

    # print("FINAL SHAPES: \n  QpropList: {} \n  testData: {}\n  classMeanList: {}".format(
    #     QpropList.shape, testData.shape, classMeanList.shape))
    #
    # print("final Qprop for class 0 {}".format(QpropList[0]))

    return testData, QpropList, classMeanList


#Zeros out the feature means, then it transposes the data
def prepPca(trainData):

    classMean = np.mean(trainData, axis = 0)
    trainData = trainData - classMean
    #print("class mean shape {}\n train data shape {}".format(classMean.shape, trainData.shape))
    # mean shape is 10304, data shape is 7, 10304

    trainData = np.transpose(trainData)

    return trainData, classMean



#resizes an image using either the height or the width and keeping the other dimension scaled
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


#Set up to run on KTH and Weizman datasets right now
def folderSetUp(actionsFolder):
    if actionsFolder == "KTH":
        maximumPhotoLimit = 10
        maximumClassLimit = 5

    else: #Weizman
        maximumPhotoLimit = 20
        maximumClassLimit = 6

    return maximumClassLimit, maximumPhotoLimit


#Make sure that the image exists, or is not just fully black
def checkImageQuality(fullFilePath, filename):
    if not filename.endswith(".jpg"):
        #print("File extension {} incorrect ".format(filename))
        return False

    if cv.countNonZero(cv.imread(fullFilePath + filename, 0)) == 0:
        print("Image is black, using next image")


    return True



def runPca(trainData, pcaInitializations):

    Qprop = l1pca.l1pca(trainData, pcaInitializations)

    Qprop = np.expand_dims(Qprop, axis = 0)
    Qmult = np.matmul(Qprop.transpose(), Qprop)

    # print("Qmult shape: {}, trainData shape : {}".format(Qmult.shape, trainData.shape))
    updatedTrainData = trainData - np.matmul(Qmult, trainData)
    # print("updatedTrainData shape: {}, trainData shape : {}".format(updatedTrainData.shape, trainData.shape))

    return Qprop, updatedTrainData


def getFolderPaths(actionsFolder, maximumClassLimit, testSet):
    #put all the class folder paths in a list
    fullFilePaths = []
    for faceClassFolder in os.listdir(actionsFolder):
        if ".DS_Store" not in faceClassFolder:
            oneLevelPath = actionsFolder + "/" + faceClassFolder
            if "Weizman" in actionsFolder:
                for innerFolder in os.listdir(oneLevelPath):
                    if ".DS_Store" not in innerFolder:
                        fullFilePaths.append(oneLevelPath + "/" + innerFolder + "/")
            else:
                for innerFolder in os.listdir(oneLevelPath):
                    if ".DS_Store" not in innerFolder:
                        fullFilePaths.append(oneLevelPath + "/" + innerFolder + "/")
    #Randomize the order of the image paths
    #random.shuffle(fullFilePaths)

    fullFilePaths.sort()
    # foldersPerClass = len(fullFilePaths) / maximumClassLimit

    #random.randint(0, math.floor(len(fullFilePaths-1)/2))
    # print("foldersPerClass: {}".format(foldersPerClass))
    # print("First {} from sorted path list: {}".format(x, fullFilePaths))
    # print("Folders from just class 1 from sorted path list: {}".format(x, int(fullFilePaths[:foldersPerClass])))

    fullFilePaths = fullFilePaths[6*(testSet):6*(testSet+1)]
    #print("Folder path list: {}".format(fullFilePaths))
    pathsToCompare = [fullFilePaths[0], fullFilePaths[1]]
    #print("full ordered path list: {}".format(fullFilePaths))

    return fullFilePaths


#small test output to check just this file

# actionsFolder = "WeizmanActions"
# testData, Qprop, classMeanList = loadActions(actionsFolder)
# print("Test Data shape: {}\nTestData: {}".format(testData.shape, testData))
# print("Qprop shape: {}".format(Qprop.shape))
# print("classMeanList shape: {}".format(classMeanList.shape))
