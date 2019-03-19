import numpy as np
import random
import cv2 as cv
import os
import random
import math
import videoImageLoader
from subspaceProjection import imageClassifier


def runActionRecognition(actionsFolder, kComponents, evaluationsPerComponent, numTests):
    print("\n###\n###\nStarting Action Recognition predicion on {} with {} components and {} evaluations per component.\n###\n###\n\n\n".format(
        actionsFolder, kComponents, evaluationsPerComponent))

    #masterFile keeps track of all of the average percentages across each component across all evaluations each in one l Return at the end
    masterFile = []
    for value in range(0, numTests):
        masterFile.append([])
    #folderPercentageCorrect keeps track of each evaluations percentage correct per component
    for testNumber in range(0, numTests):
        folderPercentageCorrect = []
        for component in range(0, kComponents):
            folderPercentageCorrect.append([])

        for i in range(0, evaluationsPerComponent):
            testData, QpropList, classMeanList = videoImageLoader.loadActions(actionsFolder, kPrincipalComponents = kComponents, testSet = testNumber)

            numberOfClasses = np.size(QpropList, 0)
            examplesPerClass = math.floor(np.size(testData, 0) / numberOfClasses)

            #keeps track of how many correctly guessed images appear per evaluation per component
            samplesCorrect = [0] * kComponents

            # print("number of classes: {}   Number of examples per class: {}".format(numberOfClasses, examplesPerClass))

            for testCounter, testImage in enumerate(testData):

                for principalComponentDimension in range(0, kComponents):
                    #pass the correct amount of dimensions for the Qprop
                    tempQpropList = QpropList[:, :principalComponentDimension +1, :]

                    #predictedClass is the returned argmin of the test image against every class subspace
                    predictedClass = imageClassifier(testImage, tempQpropList, principalComponentDimension, classMeanList)

                    #Since there is an evenly distributed number of test examples per class, the predicted class is
                    #equal to the truncated (or math.floor) value of count / examples per class
                    if math.floor(testCounter / examplesPerClass) == predictedClass:
                        samplesCorrect[principalComponentDimension] += 1

                # if testCounter % 7 == 0:
                #     print("samplesCorrect: {}".format(samplesCorrect))


            for j in range(0, len(samplesCorrect)):
                print("Samples of {} correct with {} principal components: {} OUT OF {}. Percentage: {}".format(
                    actionsFolder, j + 1, samplesCorrect[j], np.size(testData, 0), samplesCorrect[j]/np.size(testData, 0)))

                # print("adding {} to folderPercentageCorrect at position {}".format(samplesCorrect[j]/np.size(testData, 0), j))
                folderPercentageCorrect[j].append(float("%.5f"%(samplesCorrect[j]/np.size(testData, 0))))

            # print("\nAfter adding all in loop, folder % looks like : {}".format(folderPercentageCorrect))
            print("Next evaluation")
            print("_____________________________________________\n\n")
        "Done with Data Set"
        print("list of percentages {}".format(folderPercentageCorrect))
        # print("Average percentage correct with {} components over {} iterations: {}".format(
        #principalComponentDimension + 1, evaluationsPerComponent, sum(folderPercentageCorrect)/len(folderPercentageCorrect)))

        for k in range(0, len(folderPercentageCorrect)):
            masterFile[testNumber].append(sum(folderPercentageCorrect[k])/len(folderPercentageCorrect[k]))

        #averagePercentCorrect = float("%.4f"%(sum(folderPercentageCorrect)/len(folderPercentageCorrect)))
        #masterFile[testNumber].append(averagePercentCorrect)
        print("master file for this iteration: {}".format(masterFile[testNumber]))

        print()
        print("FINAL RESULTS:")
        for i in range(0, kComponents):
            print("Average percentage correct for {} components over {} iterations: {}".format(i+1, evaluationsPerComponent, masterFile[testNumber][i]))
        print("Done with file {}\n".format(actionsFolder))
        print("_____________________________________\n\n")


    return masterFile
