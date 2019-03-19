#runFaceRecognition
import numpy as np
import random
import cv2 as cv
import os
import random
import math
import imageLoader
from subspaceProjection import imageClassifier




def runFaceRecognition(facesFolder, kComponents, evaluationsPerComponent):
    print("\n###\n###\nStarting Face Recognition predicion on {} with {} components and {} evaluations per component.\n###\n###\n".format(
        facesFolder, kComponents, evaluationsPerComponent))

    masterFile = []
    folderPercentageCorrect = []
    for component in range(0, kComponents):
        folderPercentageCorrect.append([])

    for i in range(0, evaluationsPerComponent):
        testData, QpropList, classMeanList = imageLoader.loadFaces(facesFolder, kPrincipalComponents = kComponents)

        numberOfClasses = np.size(QpropList, 0)
        examplesPerClass = math.floor(np.size(testData, 0) / numberOfClasses)

        samplesCorrect = [0] * kComponents

        # print("number of classes: {}   Number of examples per class: {}".format(numberOfClasses, examplesPerClass))
        for testCounter, testImage in enumerate(testData):

            for principalComponentDimension in range(0, kComponents):

                tempQpropList = QpropList[:, :principalComponentDimension + 1, :]

                #shave off extra dims of q prop list to be the right dimension
                predictedClass = imageClassifier(testImage, tempQpropList, principalComponentDimension, classMeanList)

                if math.floor(testCounter / examplesPerClass) == predictedClass:
                    samplesCorrect[principalComponentDimension] += 1

            # if testCounter % 5 == 0:
            #     print("samplesCorrect: {}".format(samplesCorrect))


        for j in range(0, len(samplesCorrect)):
            print("Samples of {} correct with {} principal components: {} OUT OF {}. Percentage: {}".format(
                facesFolder, j + 1, samplesCorrect[j], np.size(testData, 0), samplesCorrect[j]/np.size(testData, 0)))

            # print("adding {} to folderPercentageCorrect at position {}".format(samplesCorrect[j]/np.size(testData, 0), j))
            folderPercentageCorrect[j].append(float("%.5f"%(samplesCorrect[j]/np.size(testData, 0))))

        # print("\nAfter adding all in loop, folder % looks like : {}".format(folderPercentageCorrect))
        print("Evaluation {} complete".format(i))
        print("_____________________________________________\n\n")
    "Done with Data Set"
    print("list of percentages {}".format(folderPercentageCorrect))
    # print("Average percentage correct with {} components over {} iterations: {}".format(
    #principalComponentDimension + 1, evaluationsPerComponent, sum(folderPercentageCorrect)/len(folderPercentageCorrect)))

    for k in range(0, len(folderPercentageCorrect)):
        masterFile.append(float("%.5f"%(sum(folderPercentageCorrect[k])/len(folderPercentageCorrect[k]))))

    #averagePercentCorrect = float("%.4f"%(sum(folderPercentageCorrect)/len(folderPercentageCorrect)))
    #masterFile.append(averagePercentCorrect)
    print("master file: {}".format(masterFile))

    print()
    print("FINAL RESULTS:")
    for i in range(0, kComponents):
        print("Average percentage correct for {} components over {} iterations: {}".format(i+1, evaluationsPerComponent, masterFile[i]))
    print("Done with file {}\n".format(facesFolder))
    print("_____________________________________\n\n")
    return masterFile
