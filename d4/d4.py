import numpy
import cv2

RESULT_SHAPE = (3100, 2500, 3)


# setup video file, handle reading from video, and initial setup
def run(fileName):
    previousFrame = None
    cv2.namedWindow("Display frame", cv2.WINDOW_AUTOSIZE)
    cap = cv2.VideoCapture(fileName)
    while(cap.isOpened()):
        _, frame = cap.read()
        # skip frames
        for _ in range(10):
            cap.grab()

        try:
            # just started, do some setup
            if previousFrame is None:
                # result with first frame in middle left hand side of image
                result = numpy.zeros(((RESULT_SHAPE[1], RESULT_SHAPE[0], RESULT_SHAPE[2])), dtype=numpy.uint8)
                result[1000:frame.shape[0] + 1000, 200:frame.shape[1] + 200] = frame
                previousFrame = result
                lastH = None
            # normal execution
            else:
                (result, mapping, lastH) = processFrame(previousFrame, frame, result, lastH)
                previousFrame = frame
                # less boring when we see intermediate result
                cv2.imshow('Display frame', result)
                cv2.waitKey(1)
        except cv2.error:
            break
    # cleanup
    cap.release()
    cv2.destroyAllWindows()
    return result


def processFrame(previous, current, result, lastH):
    # get homography between current and previous frame
    (matches, H, mask, mapping) = match(previous, current)
    # homography to original frame
    if lastH is not None:
        H = numpy.dot(H, lastH)
    # stitch results
    previousResult = result
    result = cv2.warpPerspective(current, H, (RESULT_SHAPE[0], RESULT_SHAPE[1]), flags=cv2.WARP_INVERSE_MAP)
    result = andImg(result, previousResult)
    return (result, mapping, H)


# ~ devoir 3
def match(left, right):
    # detecter les pts dans l'image src et cible
    method = cv2.xfeatures2d.SIFT_create()
    keypoints1, descriptors1 = method.detectAndCompute(left, None)
    keypoints2, descriptors2 = method.detectAndCompute(right, None)
    # apparier les pts avec knn
    matches = cv2.BFMatcher().knnMatch(descriptors1, descriptors2, k=2)
    # test ratio
    goodMatches = []
    for source, target in matches:
        if source.distance < 0.8 * target.distance:
            goodMatches.append(source)
    matchesMask = None
    if len(goodMatches) > 10:
        # source.queryIdx: point dans l'image source; source.trainIdx: point apparié dans l'image cible
        sourcePoints = __cvMatReshape([keypoints1[source.queryIdx].pt for source in goodMatches])
        destinationPoints = __cvMatReshape([keypoints2[source.trainIdx].pt for source in goodMatches])
        # trouver l'homographie
        retval, mask = cv2.findHomography(sourcePoints, destinationPoints, cv2.RANSAC)
        matchesMask = mask.ravel().tolist()
        mapping = cv2.drawMatches(left, keypoints1, right, keypoints2, goodMatches, None, matchesMask=matchesMask)
        return (goodMatches, retval, mask, mapping)


# stitch two images
def andImg(backgroundImage, foregroundImage):
    rows, cols, _ = foregroundImage.shape
    roi = backgroundImage[0:rows, 0:cols]
    foregroundImagegray = cv2.cvtColor(foregroundImage, cv2.COLOR_BGR2GRAY)
    ret, mask = cv2.threshold(foregroundImagegray, 1, 255, cv2.THRESH_BINARY)
    mask_inv = cv2.bitwise_not(mask)
    backgroundImage_bg = cv2.bitwise_and(roi, roi, mask=mask_inv)
    foregroundImage_fg = cv2.bitwise_and(foregroundImage, foregroundImage, mask=mask)
    dst = cv2.add(backgroundImage_bg, foregroundImage_fg)
    backgroundImage[0:rows, 0:cols] = dst
    return backgroundImage


def __cvMatReshape(array):
    return numpy.float32(array).reshape(-1, 1, 2)

# MAIN <- starts here
try:
    result = run('mur.mp4')
except:
    pass
finally:
    cv2.imwrite('mur.jpeg', result)
