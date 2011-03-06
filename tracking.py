from tiffstack import TiffStack
import cv
from image_tools import *
from scipy import ndimage
import pylab
import pymorph
import unittest
import random

file_name = "data/640R mcciek  laser60p and 12 days003.nd2 - C=0.tif"
r = 10.0

class OpenCV():
    
    prev_points = None
    points = None
    features = None
    prev_features = None
    file = None
    grey = None
    prev_grey = None
    image = None
    prev_image = None
    win_name = None
    grey_arr = None
    arr = None
    tmp = None
    
    def __init__(self, name="opencv"):
        self.win_name = name
        
    def load(self, file):
        self.file = file
    
    def loadTiffStack(self):
        self.stack = TiffStack(self.file)
    
    def getTiff(self, ix):
        return self.stack.__getitem__(ix)
    
    def setCurrIm(self, im):
        self.image = im
    
    def expandCurrIm(self):
        self.image, self.tmp = expand(self.image)
        self.arr = cv2array(self.image)

    def setGrey(self):
        self.grey = CreateImage(GetSize(self.image), 8, 1)
        Copy(self.image, self.grey)
        self.grey_arr = cv2array(self.grey)
        self.image = CreateImage(GetSize(self.image), 8, 3)
        CvtColor(self.grey, self.image, CV_GRAY2BGR)
        
    def setFeatures(self, num_features = 100):
        self.setGrey()
        self.features = getFeatures(self.grey, num_features)
    
    def save(self, count):
        SaveImage("images/frame%d.png"%count, self.image)

    def setQuickFeatures(self):
        self.setGrey()
        self.features = getQuickFeatures(self.grey_arr)

    def drawFeatures(self, features=None):
        if not features: features = self.features
        self.image = drawFeatures(self.image, features)
    
    def drawPoints(self):
        self.image = drawPoints(self.tmp, self.points)
        
    def show(self, wait=None):
        cv.NamedWindow(self.win_name)
        cv.ShowImage(self.win_name, self.image)
        if not wait:
            cv.WaitKey()
        else:
            cv.WaitKey(wait)
    
    def step(self):
        self.prev_features = self.features
        self.prev_points = self.points
        self.prev_grey = self.grey
        self.prev_image = self.image
    
    def thresh(self, min=110, max=255):
        self.image = thresh(self.image, min, max)

    def removeOverlaps(self):
        seen = []
        for i in self.points:
            already_seen = False
            for j in seen:
                if already_seen or dist((j.x, j.y), i) < i.radius:
                    already_seen = True
            if not already_seen:
                seen.append(i)
        self.points = seen
    
    def correctPoints(self, features):
        to_remove = []
        im = self.image
        t = mahotas.thresholding.otsu(self.grey_arr)
        for i in self.points:
            bright = []
            mindist = i.radius*2

            mind = dist(features[0], i)
            mini = features[0]

            for j in features:
                if dist(j, i) < mind:
                    mind = dist(j,i)
                    mini = j
            if mind < mindist:
                i.x = mini[0]
                i.y = mini[1]

            i.x = int(max(min(i.x, im.width),0))
            i.y = int(max(min(i.y, im.height),0))
            
            x_0 = int(max(min(i.x-i.radius, im.width),0))
            x_1 = int(max(min(i.x+i.radius, im.width),0))
            y_0 = int(max(min(i.y-i.radius, im.height),0))
            y_1 = int(max(min(i.y+i.radius, im.height),0))

            brightness = []
            for x in xrange(x_0,x_1):
                for y in xrange(y_0,y_1):
                    pixel = self.grey_arr[x][y]
                    brightness.append(pixel)
            bright = np.sum(brightness)
#            if np.max(brightness) > t:
            i.brightness = bright
#            else: to_remove.append(i)
        for i in to_remove:
            self.points.remove(i)
        self.removeOverlaps()
                
    def getPoints(self):
        global r
        self.points = []
        prev_seen = []
        if not self.prev_points:
            self.prev_points = [Point(i[0], i[1]) for i in self.prev_features]

        for i in self.features:
            min = dist(i, self.prev_points[0])
            mini = self.prev_points[0]
            for j in self.prev_points:
                if dist(i,j) < min:
                    min = dist(i,j)
                    mini = j
            if min < r * 4 and not mini.next:
                prev_seen.append(mini)
                point = Point(i[0], i[1])
                point.prev = mini
                point.color = mini.color
                mini.next = point
                self.points.append(point)

    def opticalFlow(self):
        self.features = calcOptimalFlow(self.prev_grey, self.grey, self.prev_features)
        self.points = []

    def plotPoints(self,total_count):
        plots = []
        pylab.clf()
        for i in self.points:
            plot = []
            origin = i
            while origin.prev:
                origin = origin.prev
            count = 1
            while origin:
                plot.append(origin.brightness)
                origin = origin.next
                count = count+1
            #print plot
            pylab.plot(plot)
            plots.append(plot)
        pylab.savefig("graphs/graph%d.png"%total_count)
        np.save("plots/plot%d.dat"%total_count, plots)
            
            
def dist(x,y):
    a = x[0]-y.x
    b = x[1]-y.y
    return Sqrt(a*a+b*b)

class Point():
    def __init__(self, x, y):
        global r
        self.x = x
        self.y = y
        self.prev = None
        self.next = None
        self.brightness = 0
        self.radius = r
        self.color = (random.randint(0,256),
                      random.randint(0,256),
                      random.randint(0,256), 0)
        
        
class Test(unittest.TestCase):
    
    opencv = OpenCV()
    global file_name
    def testLoad(self):
        self.opencv.load(file_name)
        self.opencv.loadTiffStack()
        self.opencv.setCurrIm(self.opencv.getTiff(1))
        self.opencv.show()
    
    def testExpand(self):
        self.opencv.load(file_name)
        self.opencv.loadTiffStack()
        self.opencv.setCurrIm(self.opencv.getTiff(1))
        self.opencv.expandCurrIm()
        self.opencv.show()
    
    def testGetFeatures(self):
        self.opencv.load(file_name)
        self.opencv.loadTiffStack()
        self.opencv.setCurrIm(self.opencv.getTiff(1))
        self.opencv.expandCurrIm()
        self.opencv.setFeatures()
        self.opencv.drawFeatures()
        self.opencv.show()
        
    def testGetThresh(self):
        self.opencv.load(file_name)
        self.opencv.loadTiffStack()
        self.opencv.setCurrIm(self.opencv.getTiff(1))
        self.opencv.expandCurrIm()
        self.opencv.thresh()
        self.opencv.setFeatures()
        self.opencv.drawFeatures()
        self.opencv.show()
    
    def testTrackImage(self):
        self.opencv.load(file_name)
        self.opencv.loadTiffStack()
        self.opencv.setCurrIm(self.opencv.getTiff(1))
        self.opencv.expandCurrIm()
        self.opencv.setFeatures()
        self.opencv.step()
        self.opencv.setCurrIm(self.opencv.getTiff(2))
        self.opencv.expandCurrIm()
        self.opencv.setFeatures()
        self.opencv.opticalFlow()
        self.opencv.drawFeatures()
        self.opencv.show()
    
    def testTrackImages(self):
        self.opencv.load(file_name)
        self.opencv.loadTiffStack()
        i = self.opencv.getTiff(1)
        self.opencv.setCurrIm(i)
        self.opencv.expandCurrIm()
        self.opencv.setQuickFeatures()
        self.opencv.drawFeatures()
#        self.opencv.show()
#        self.opencv.step()
        tmp = OpenCV("opencv2")
        count = 1
        for i in self.opencv.stack:
            if i is not None:
                self.opencv.step()
                self.opencv.setCurrIm(i)
                self.opencv.expandCurrIm()
                #tmp.setCurrIm(i)
                #tmp.expandCurrIm()
                #tmp.setFeatures(100)
                #self.opencv.thresh()
                
#                self.opencv.setFeatures(100)
                self.opencv.setQuickFeatures()
                self.opencv.opticalFlow()
                self.opencv.getPoints()
                self.opencv.correctPoints(getQuickFeatures(self.opencv.arr))
                self.opencv.drawPoints()
                count = count+1
                print "frame %d" % count
                
                self.opencv.save(count)

                self.opencv.plotPoints(count)
                self.opencv.show(100)

    def testPlot(self):
        data = [(i,o*o) for i,o in enumerate(xrange(0,100))]
        #pylab.plot(data)
        #pylab.show()

    def testAddFeature(self):
        self.opencv.load(file = "data/640R mcciek  laser60p and 12 days003.nd2 - C=0.tif")
        self.opencv.loadTiffStack()
        i = self.opencv.getTiff(1)
        self.opencv.setCurrIm(i)
        SetMouseCallback ('opencv', on_mouse, None)  

if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
