
import os, sys, glob, pathlib, imutils, shutil, json
from tqdm import tqdm
from uuid import uuid4
from argparse import ArgumentParser
import cv2
import numpy as np
from matplotlib import pyplot as plt
from shapely.geometry import Polygon


def empty():
	pass

def getContours(img,preprocessedImg, areaMin):
    '''
    method to extract the contours
    '''
    contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    for cnt in contours:
        area = cv2.contourArea(cnt)

        if isChoseParameters:
            areaMin = cv2.getTrackbarPos("AreaMin", "Parameters")

        # if the area is more than the minimium
        if area > areaMin:

            # Draw the contours on the image
            cv2.drawContours(preprocessedImg, cnt, -1, (255, 0, 255), 7)

            # Apporximate the contours
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)

            # Getting the bounding box
            x , y , w, h = cv2.boundingRect(approx)

            if verbose == 2 :
                # Draw a rectangle around the output
                cv2.rectangle(preprocessedImg, (x , y ), (x + w , y + h ), (0, 255, 0), 5)
                cv2.putText(preprocessedImg, "Points: " + str(len(approx)), (x + w + 20, y + 20), cv2.FONT_HERSHEY_COMPLEX, .7,
                            (0, 255, 0), 2)
                cv2.putText(preprocessedImg, "Area: " + str(int(area)), (x + w + 20, y + 45), cv2.FONT_HERSHEY_COMPLEX, 0.7,
                            (0, 255, 0), 2)

            return {"x" : x,
                    "y" : y,
                    "w" : w,
                    "h" : h
                    }

def stackImages(scale,imgArray):
    '''
    method to stack all the images and easily display them
    '''
    rows = len(imgArray)
    cols = len(imgArray[0])
    rowsAvailable = isinstance(imgArray[0], list)
    width = imgArray[0][0].shape[1]
    height = imgArray[0][0].shape[0]
    if rowsAvailable:
        for x in range ( 0, rows):
            for y in range(0, cols):
                if imgArray[x][y].shape[:2] == imgArray[0][0].shape [:2]:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (0, 0), None, scale, scale)
                else:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (imgArray[0][0].shape[1], imgArray[0][0].shape[0]), None, scale, scale)
                if len(imgArray[x][y].shape) == 2: imgArray[x][y]= cv2.cvtColor( imgArray[x][y], cv2.COLOR_GRAY2BGR)
        imageBlank = np.zeros((height, width, 3), np.uint8)
        hor = [imageBlank]*rows
        hor_con = [imageBlank]*rows
        for x in range(0, rows):
            hor[x] = np.hstack(imgArray[x])
        ver = np.vstack(hor)
    else:
        for x in range(0, rows):
            if imgArray[x].shape[:2] == imgArray[0].shape[:2]:
                imgArray[x] = cv2.resize(imgArray[x], (0, 0), None, scale, scale)
            else:
                imgArray[x] = cv2.resize(imgArray[x], (imgArray[0].shape[1], imgArray[0].shape[0]), None,scale, scale)
            if len(imgArray[x].shape) == 2: imgArray[x] = cv2.cvtColor(imgArray[x], cv2.COLOR_GRAY2BGR)
        hor= np.hstack(imgArray)
        ver = hor
    return ver

def intializeParameterWindow() :
    '''
    method to create the parameter window
    '''
    cv2.namedWindow("Parameters")
    cv2.resizeWindow("Parameters",640, 240)
    cv2.createTrackbar("Threshold1", "Parameters", 26, 244, empty)
    cv2.createTrackbar("Threshold2", "Parameters", 90, 255, empty)
    cv2.createTrackbar("AreaMin", "Parameters", 150000, 550000, empty)

def processing(path, currency, name):
    '''
    method for the processing
    '''
    cap = cv2.VideoCapture(path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    for i in tqdm(range(0, total, 4)):

        # Read each frame
        sucess, img = cap.read()

        if not sucess:
            break

        img_final = img.copy()

        # Preprocessing for canny edge detector
        preprocessedImg, mask = preprocessing_for_canny(img)

        # Get the canny edge image
        imgCanny = cv2.Canny(preprocessedImg, threshold1, threshold2)

        # Dialate the image to expand the enhance the edges
        kernal = np.ones((7,7))
        imgDil = cv2.dilate(imgCanny, kernal, iterations=1)

        # Get the contours
        # boundingBox = getContours(imgDil, preprocessedImg, areaMin)

        img_final[:,:,0] = img_final[:,:,0] * mask
        img_final[:,:,1] = img_final[:,:,1] * mask
        img_final[:,:,2] = img_final[:,:,2] * mask


        b_channel, g_channel, r_channel = cv2.split(img_final)
        alpha_channel = (mask*255).astype(np.uint8)
        imgbgra = cv2.merge((b_channel, g_channel, r_channel, alpha_channel))
        fimage = cv2.medianBlur(imgbgra, 5)

        # Generate the polygons
        label = get_polygons(fimage)

        # Write the images to file
        with open('{}/images/{}/labels/{}_{}.json'.format(datafolder,currency,name,i), 'w') as f:
            json.dump(label, f)
        cv2.imwrite('{}/images/{}/images/{}_{}.png'.format(datafolder,currency,name,i), fimage)

        if verbose == 2:
            # Display the images
            imgStack = stackImages(0.5, ([img, imgCanny], [imgDil, preprocessedImg]))
            cv2.imshow("Result", imgStack)

        # Exit from the window
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

def preprocessing_for_canny(img) :
    # Convert BGR to HSV
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    if isGreen :
        # mask for green
        mask = cv2.inRange(hsv, (36, 25, 25), (100, 255,255))
    else :
        # mask for blue
        mask = cv2.inRange(hsv, (90, 25, 25), (140, 255,255))

    ## slice the mask
    imask = mask == 0
    mask_channel = np.zeros_like(img, np.uint8)
    mask_channel[imask] = img[imask]

    # converting the image from HSV to RGB
    img_sub_green = cv2.cvtColor(mask_channel, cv2.COLOR_HSV2BGR)

    # Take copy of the frame
    preprocessedImg = img_sub_green.copy()

    # Add gaussian blur
    imgBlur = cv2.GaussianBlur(preprocessedImg, (9,9), 1)

    # Convert the image to gray
    imgGray = cv2.cvtColor(imgBlur, cv2.COLOR_BGR2GRAY)

    if isChoseParameters:
        # Get the values from the tracker bar
        threshold1 = cv2.getTrackbarPos("Threshold1", "Parameters")
        threshold2 = cv2.getTrackbarPos("Threshold2", "Parameters")

    return preprocessedImg, imask


def get_polygons(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # find contours in the thresholded image
    cnts = cv2.findContours(img.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)

    # cnts = imutils.grab_contours(contours)
    cnts = sorted(cnts, key = cv2.contourArea, reverse = True)[:10]

    cnt = cnts[0]
    # print(cnt.shape)
    points, _, _ = cnt.shape

    ncnt = np.reshape(cnt, (points, 2))

    polygon = Polygon(ncnt)

    simplepolygon = polygon.simplify(1.0, preserve_topology=False)

    if simplepolygon.type == 'MultiPolygon':
        points = []
        for polygon in simplepolygon:
            subpoints = list(polygon.exterior.coords)
            for subpoint in subpoints:
                points.append(subpoint)

    elif simplepolygon.type == 'Polygon':
        points = list(simplepolygon.exterior.coords)

    label = {
        'points' : points
    }

    return label


def main():
    if isChoseParameters:
        intializeParameterWindow()

    for currency in currencies:
        for afile in tqdm(sorted(glob.glob('{}/videos/{}/*.MOV'.format(datafolder, currency)))):
            name = pathlib.Path(afile).stem
            processing(afile,currency,name)

def setup():

    try:
        # Remove the folder
        shutil.rmtree("{}/images/".format(datafolder))
    except FileNotFoundError:
        print("Creating new folder.")

    os.mkdir('{}/images'.format(datafolder))
    for currency in currencies:
        os.mkdir('{}/images/{}'.format(datafolder,currency))
        os.mkdir('{}/images/{}/images'.format(datafolder,currency))
        os.mkdir('{}/images/{}/labels'.format(datafolder,currency))




if __name__ == "__main__" :

    # parse arguments
    parser = ArgumentParser()
    parser.add_argument("-v", "-verbose", dest="verbose", default=0,
                        help="control the output")
    parser.add_argument("-isgreen", dest="isgreen", default=False,
                        help="if screen is green")
    parser.add_argument("-control", dest="control", default=False,
                        help="control the parameters")
    args = parser.parse_args()


    verbose = int(args.verbose)
    isGreen = args.isgreen
    isChoseParameters = args.control

    # Script parameters
    threshold1 = 26
    threshold2 = 90
    areaMin = 150000
    datafolder = "data"
    currencies = ["RM1","RM5","RM10","RM20","RM50","RM100"]

    setup()

    main()
