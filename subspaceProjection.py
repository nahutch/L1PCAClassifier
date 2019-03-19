import numpy as np


def subspaceProjection(testImage, Qprop, classMean):

    dataZeroedClassMean = testImage - classMean  #sample % 3 if orl, %8 if yale.  mod by images/class
    QpropMult = np.matmul(Qprop.transpose(), Qprop)

    #print("dataZeroedClassMean: {}".format(dataZeroedClassMean))
    #print()
    #print()
    return np.linalg.norm(dataZeroedClassMean - (np.matmul(QpropMult, dataZeroedClassMean)), 2)




def imageClassifier(testImage, Qprop, principalComponentDimension, classMeanList):

    numberOfClasses = np.size(Qprop, 0)
    classProjectionValues = np.zeros((numberOfClasses, 1))

    for QpropCounter, classQprop in enumerate(Qprop):
        classProjectionValues[QpropCounter] = subspaceProjection(testImage, classQprop[:principalComponentDimension + 1], classMeanList[QpropCounter])
        #print("ClassProjectionValue at iteration {} : {}".format(QpropCounter, classProjectionValues))

    return np.argmin(classProjectionValues)
