import cv2, glob, pathlib, json
import numpy as np
from tqdm import tqdm
from shapely.geometry import Polygon
import imutils
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def get_polygons(folder, currency):

    for apath in tqdm(sorted(glob.glob('{}/{}/images/*.png'.format(folder,currency)))):
        name = pathlib.Path(apath).stem

        img = cv2.imread('{}'.format(apath), 0)
        img2 = cv2.imread('{}'.format(apath), cv2.IMREAD_UNCHANGED)

        img = cv2.medianBlur(img, 5)
        # contours, hierarchy =   cv2.findContours(img.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

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

        with open('{}/{}/labels/{}.json'.format(folder,currency,name), 'w') as f:
            json.dump(label, f)

        # img2 = cv2.drawContours(img2, [t], 0, (0,255,0), 3)
        # cv2.imshow('Contours', img2)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        # img2 = cv2.drawContours(img2.copy(), contours, -1, (0,255,0), 3)

        # cv2.imwrite('{}/render/{}.png'.format(folder,name), img2)



if __name__ == "__main__":
    get_polygons("data", "RM50")