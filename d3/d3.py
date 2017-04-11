from matplotlib import pyplot as plt
from os import listdir
import numpy
import cv2


def findObjectInImage(sourceImagePath, targetImagePath):
    sourceImage = cv2.imread(sourceImagePath, 0)
    targetImage = cv2.imread(targetImagePath, 0)
    # detecter les pts dans l'image src et cible
    orb = cv2.ORB_create()
    keypoints1, descriptors1 = orb.detectAndCompute(sourceImage, None)
    keypoints2, descriptors2 = orb.detectAndCompute(targetImage, None)
    # apparier les pts avec knn
    matches = cv2.BFMatcher().knnMatch(descriptors1, descriptors2, k=2)
    # test ratio
    goodMatches = []
    for source, target in matches:
        if source.distance < 0.8 * target.distance:
            goodMatches.append(source)
    matchesMask = None
    if len(goodMatches) > 10:
        # l'objet prend toute l'image source donc les coins de l'objet sont les coins de l'image
        height, width = sourceImage.shape
        corners = cvMatReshape([[0, 0], [0, height - 1], [width - 1, height - 1], [width - 1, 0]])
        # source.queryIdx: point dans l'image source; source.trainIdx: point apparié dans l'image cible
        sourcePoints = cvMatReshape([keypoints1[source.queryIdx].pt for source in goodMatches])
        destinationPoints = cvMatReshape([keypoints2[source.trainIdx].pt for source in goodMatches])
        # trouver l'homographie
        retval, mask = cv2.findHomography(sourcePoints, destinationPoints, cv2.RANSAC)
        matchesMask = mask.ravel().tolist()
        destinationCorners = cv2.perspectiveTransform(corners, retval)
        # dessiner le contour de l'objet dans l'image cible
        targetImage = cv2.polylines(targetImage, [numpy.int32(destinationCorners)], True, 0, 5)
    # dessiner les appariements
    mapping = cv2.drawMatches(sourceImage, keypoints1, targetImage, keypoints2, goodMatches, None, matchesMask=matchesMask)
    plt.imshow(mapping)
    plt.show()


def cvMatReshape(array):
    return numpy.float32(array).reshape(-1, 1, 2)

# commencer ici
for targetImage in listdir('camions/'):
    findObjectInImage('poste.jpg', 'camions/' + targetImage)
