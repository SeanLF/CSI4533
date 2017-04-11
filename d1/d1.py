import cv2
import numpy as np


def findBall(imagePath, lowerHSV, upperHSV):
    # lire l'image
    image = cv2.imread(imagePath)
    # changer le domaine de couleur (HSV)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # trouver le masque qui selectionne toutes les couleurs dans notre plage de HSV
    mask = cv2.inRange(hsv, np.array(lowerHSV), np.array(upperHSV))

    # faire l'ouverture du masque
    kernel = np.ones((2, 2), np.uint8)
    morphedMask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    # trouver tous les contours possible
    _, contours, _ = cv2.findContours(morphedMask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    bestBoundingRect = {'minRatio': 2, 'values': None}
    # pour chaque contour trouvé
    for contour in contours:
        # trouver la boîte englobante du contour
        x, y, width, height = cv2.boundingRect(contour)
        # si le contour est relativement carré et mesure entre 35px et 75px de côté
        if (35 <= width <= 75) and (35 <= height <= 75) and (height * 0.8 <= width <= height * 1.2):
            currentRatio = max(height, width) / min(height, width)
            # prendre la boîte englobante la plus carré
            if currentRatio < bestBoundingRect['minRatio']:
                bestBoundingRect = {'values': [x, y, width, height], 'minRatio': currentRatio}

    # dessiner le rectangle
    if bestBoundingRect['values'] is not None:
        x, y, width, height = bestBoundingRect['values']
        rectangle = cv2.rectangle(image, (x, y), (x + width, y + height), (0, 255, 0), 2)

    # montrer l'image avec la boîte englobante et attendre que l'utilisateur frappe une touche pour fermer la fenêtre
    cv2.namedWindow("bounded", cv2.WINDOW_NORMAL)
    cv2.imshow('bounded', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.waitKey(1)

for ballNumber in range(1, 4):
    # trouver la couleur principale de chaque balle (domaine HSV)
    hsv = cv2.cvtColor(cv2.imread('images/ball' + str(ballNumber) + '.jpg'), cv2.COLOR_BGR2HSV)
    maxHue = np.argmax(cv2.calcHist([hsv], [0], None, [180], [0, 180]))
    # trouver chaque balle dans chaque image
    for imageNumber in range(1, 4):
        findBall('images/image' + str(imageNumber) + '.jpg', [maxHue - 7, 50, 50], [maxHue + 13, 255, 255])
