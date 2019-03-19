#main.py
import cv2 as cv
import numpy as np    # for mathematical operations
import matplotlib.pyplot as plt

from runFaceRecognition import runFaceRecognition
from runActionRecognition import runActionRecognition
from videoSetUp import populateActionVideos

#RUN THESE LINES FIRST TO POPULATE THE FOLDERS IN THE DIRECTORY WITH THE SAVED MOTION IMAGE HISTORY
videoClassFoldersWiezman = ["Weizman/run", "Weizman/walk", "Weizman/wave1", "Weizman/wave2", "Weizman/jump", "Weizman/bend"]
for vidClass in videoClassFoldersWiezman:
    populateActionVideos(vidClass)

videoClassFoldersKTH = ["KTH/running", "KTH/walking", "KTH/handwaving", "KTH/handclapping", "KTH/boxing"]
for vidClass in videoClassFoldersKTH:
    populateActionVideos(vidClass)


#hyperparameters to change
kComponents = 6
evaluationsPerComponent = 40

#num tests will try numTests amount of different combinations of MHI_DURATION and THRESHOLD from the video folders
numTests = 9

masterOrl = runFaceRecognition("orl_faces", kComponents, evaluationsPerComponent)
masterYale = runFaceRecognition("CroppedYale", kComponents, evaluationsPerComponent)
print("masterOrl: {}".format(masterOrl))
print("masterYale: {}".format(masterYale))

# masterKTH = runActionRecognition("KTHActions", kComponents, evaluationsPerComponent, numTests)
masterWeizman = runActionRecognition("WeizmanActions", kComponents, evaluationsPerComponent, numTests)

# print("master kth: ".format(masterKTH))
print("masterWeizman: {}".format(masterWeizman))
print("masterOrl: {}".format(masterOrl))
print("masterYale: {}".format(masterYale))



componentLabel = []
for i in range(1, kComponents+1):
    componentLabel.append(i)

#inverts the percent correct to % error
masterWeizman[0][:] = [1 - x for x in masterWeizman[0]]
masterWeizman[1][:] = [1 - x for x in masterWeizman[1]]
masterWeizman[2][:] = [1 - x for x in masterWeizman[2]]
masterWeizman[3][:] = [1 - x for x in masterWeizman[3]]
masterWeizman[4][:] = [1 - x for x in masterWeizman[4]]
masterWeizman[5][:] = [1 - x for x in masterWeizman[5]]
masterWeizman[6][:] = [1 - x for x in masterWeizman[6]]
masterWeizman[7][:] = [1 - x for x in masterWeizman[7]]
masterWeizman[8][:] = [1 - x for x in masterWeizman[8]]

plt.figure()
plt.plot(componentLabel, masterWeizman[0], "ro", label = "mhi15, thresh15")
plt.plot(componentLabel, masterWeizman[1], "bo", label = "mhi15, thersh45")
plt.plot(componentLabel, masterWeizman[2], "go", label = "mhi15, thresh75")
plt.ylabel("Average Error Level (%)")
plt.xlabel("number of components")
plt.legend(loc = "upper right")
plt.axis([0, 6, 0, 1])
plt.show()

plt.figure()
plt.plot(componentLabel, masterWeizman[3], "ro", label = "mhi20, thresh15")
plt.plot(componentLabel, masterWeizman[4], "bo", label = "mhi20, thersh45")
plt.plot(componentLabel, masterWeizman[5], "go", label = "mhi20, thresh75")
plt.ylabel("Average Error Level (%)")
plt.xlabel("number of components")
plt.legend(loc = "upper right")
plt.axis([0, 6, 0, 1])
plt.show()

plt.figure()
plt.plot(componentLabel, masterWeizman[6], "ro", label = "mhi25, thresh15")
plt.plot(componentLabel, masterWeizman[7], "bo", label = "mhi25, thersh45")
plt.plot(componentLabel, masterWeizman[8], "go", label = "mhi25, thresh75")
plt.ylabel("Average Error Level (%)")
plt.xlabel("number of components")
plt.legend(loc = "upper right")
plt.axis([0, 6, 0, 1])
plt.show()


facesComponentLabel = []
for i in range(1, kComponents + 1):
    facesComponentLabel.append(i)

masterYale[:] = [1 - x for x in masterYale]
masterOrl[:] = [1 - x for x in masterOrl]

plt.figure()
plt.plot(facesComponentLabel, masterOrl, "ro", label = "orl dataset")
plt.plot(facesComponentLabel, masterYale, "bo", label = "Yale Dataset")
plt.ylabel("Average Error Level (%)")
plt.xlabel("Number of Principal Components of per class representation")
plt.legend(loc = "upper right")
plt.axis([0, 6, 0, 1])
plt.show()

'''
componentLabel = []
for i in range(1, kComponents+1):
    componentLabel.append(i)

masterKTH = [[0.671431, 0.707145, 0.707145, 0.740476, 0.754761, 0.773808], [0.590478, 0.642859, 0.688098, 0.69524, 0.707145, 0.728574]]

masterKTH[0][:] = [1 - x for x in masterKTH[0]]
masterKTH[1][:] = [1 - x for x in masterKTH[1]]
plt.figure()
plt.plot(componentLabel, masterKTH[0], "ro", label = "mhi25, thresh15")
plt.plot(componentLabel, masterKTH[1], "bo", label = "mhi25, thresh 45")
plt.ylabel("Average Error Level (%)")
plt.xlabel("number of components")
plt.legend(loc = "upper right")
plt.axis([0, 6, 0, 1])
plt.show()
'''
