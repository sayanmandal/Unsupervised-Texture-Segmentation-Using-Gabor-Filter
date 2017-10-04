
import math
import sklearn.cluster as clstr
import cv2
import numpy as np
from PIL import Image, ImageOps, ImageDraw
import os, glob
import matplotlib.pyplot as pyplt
import scipy.cluster.vq as vq
import argparse
import glob

# We can specify these if need be.
brodatz = "D:\\ImageProcessing\\project\\OriginalBrodatz\\"
concatOut = "D:\\ImageProcessing\\project\\concat.png"

# This is the function that checks boundaries when performing spatial convolution.
def getRanges_for_window_with_adjust(row, col, height, width, W):

    mRange = []
    nRange = []

    mRange.append(0)
    mRange.append(W-1)

    nRange.append(0)
    nRange.append(W-1)

    initm = int(round(row - math.floor(W / 2)))
    initn = int(round(col - math.floor(W / 2)))

    if (initm < 0):
        mRange[1] += initm
        initm = 0

    if (initn < 0):
        nRange[1] += initn
        initn = 0

    if(initm + mRange[1] > (height - 1)):
        diff = ((initm + mRange[1]) - (height - 1))
        mRange[1] -= diff

    if(initn + nRange[1] > (width-1)):
        diff = ((initn + nRange[1]) - (width - 1))
        nRange[1] -= diff

    windowHeight = mRange[1] - mRange[0]
    windowWidth = nRange[1] - nRange[0]

    return int(round(windowHeight)), int(round(windowWidth)), int(round(initm)), int(round(initn))

# Used to normalize data before clustering occurs.
# Whiten sets the variance to be 1 (unit variance),
# spatial weighting also takes place here.
# The mean can be subtracted if specified by the implementation.
def normalizeData(featureVectors, setMeanToZero, spatialWeight=1):

    means = []
    for col in range(0, len(featureVectors[0])):
        colMean = 0
        for row in range(0, len(featureVectors)):
            colMean += featureVectors[row][col]
        colMean /= len(featureVectors)
        means.append(colMean)

    for col in range(2, len(featureVectors[0])):
        for row in range(0, len(featureVectors)):
            featureVectors[row][col] -= means[col]
    copy = vq.whiten(featureVectors)
    if (setMeanToZero):
        for row in range(0, len(featureVectors)):
            for col in range(0, len(featureVectors[0])):
                copy[row][col] -= means[col]

    for row in range(0, len(featureVectors)):
        copy[row][0] *= spatialWeight
        copy[row][1] *= spatialWeight

    return copy

# Create the feature vectors and add in row and column data
def constructFeatureVectors(featureImages, img):

    featureVectors = []
    height, width = img.shape
    for row in range(height):
        for col in range(width):
            featureVector = []
            featureVector.append(row)
            featureVector.append(col)
            for featureImage in featureImages:
                featureVector.append(featureImage[row][col])
            featureVectors.append(featureVector)

    return featureVectors

# An extra function if we are looking to save our feature vectors for later
def printFeatureVectors(outDir, featureVectors):

    f = open(outDir, 'w')
    for vector in featureVectors:
        for item in vector:
            f.write(str(item) + " ")
        f.write("\n")
    f.close()

# If we want to read in some feature vectors instead of creating them.
def readInFeatureVectorsFromFile(dir):
    list = [line.rstrip('\n') for line in open(dir)]
    list = [i.split() for i in list]
    newList = []
    for row in list:
        newRow = []
        for item in row:
            floatitem = float(item)
            newRow.append(floatitem)
        newList.append(newRow)

    return newList

# Print the intermediate results before clustering occurs
def printFeatureImages(featureImages, naming, printlocation):

    i =0
    for image in featureImages:
        # Normalize to intensity values
        imageToPrint = cv2.normalize(image, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        cv2.imwrite(printlocation + "\\" + naming + str(i) + ".png", imageToPrint)
        i+=1

# Print the final result, the user can also choose to make the output grey
def printClassifiedImage(labels, k, img, outdir, greyOutput):

    if(greyOutput):
        labels = labels.reshape(img.shape)
        for row in range(0, len(labels)):
            for col in range(0, len(labels[0])):
                outputIntensity = (255/k)*labels[row][col]
                labels[row][col] = outputIntensity
        cv2.imwrite(outdir, labels.reshape(img.shape))
    else:
        pyplt.imsave(outdir, labels.reshape(img.shape))

# Call the k means algorithm for classification
def clusterFeatureVectors(featureVectors, k):

    kmeans = clstr.KMeans(n_clusters=k)
    kmeans.fit(featureVectors)
    labels = kmeans.labels_

    return labels

# To clean up old filter and feature images if the user chose to print them.
def deleteExistingSubResults(outputPath):
    for filename in os.listdir(outputPath):
        if (filename.startswith("filter") or filename.startswith("feature")):
            os.remove(filename)

# Checks user input (i.e. cannot have a negative mask size value)
def check_positive_int(n):
    int_n = int(n)
    if int_n < 0:
         raise argparse.ArgumentTypeError("%s is negative" % n)
    return int_n

# Checks user input (i.e. cannot have a negative weighting value)
def check_positive_float(n):
    float_n = float(n)
    if float_n < 0:
         raise argparse.ArgumentTypeError("%s is negative " % n)
    return float_n

#--------------------------------------------------------------------------
# All of the functions below were left here to demonstrate how I went about
# cropping the input images. I left them here, in the case that Brodatz
# textures were downloaded and cropped as new input images.
#--------------------------------------------------------------------------

def cropTexture(x_offset, Y_offset, width, height, inDir, outDir):

    box = (x_offset, Y_offset, width, height)
    image = Image.open(inDir)
    crop = image.crop(box)
    crop.save(outDir, "PNG")

def deleteCroppedImages():
    for filename in glob.glob(brodatz + "*crop*"):
        os.remove(filename)

def concatentationOfBrodatzTexturesIntoRows(pathsToImages, outdir, axisType):
    images = []
    for thisImage in pathsToImages:
        images.append(cv2.imread(thisImage, cv2.CV_LOAD_IMAGE_GRAYSCALE))
    cv2.imwrite(outdir, np.concatenate(images, axis=axisType))

    outimg = cv2.imread(outdir, cv2.CV_LOAD_IMAGE_GRAYSCALE)

    return outimg

def createGrid(listOfBrodatzInts, outName, howManyPerRow):

    listOfRowOutputs = []
    for i in range(len(listOfBrodatzInts)):
        brodatzCropInput = brodatz + "D" + str(listOfBrodatzInts[i]) + ".png"
        brodatzCropOutput = brodatz + "cropD" + str(listOfBrodatzInts[i]) + ".png"
        # 128x128 crops, in order to generate a 512x512 image
        cropTexture(256, 256, 384, 384, brodatzCropInput, brodatzCropOutput)
        listOfRowOutputs.append(brodatzCropOutput)
    subOuts = [listOfRowOutputs[x:x + howManyPerRow] for x in xrange(0,len(listOfRowOutputs), howManyPerRow)]
    dests = []
    for i in range(len(subOuts)):
        dest = brodatz + "cropRow" + str(i) + ".png"
        dests.append(dest)
        concatentationOfBrodatzTexturesIntoRows(subOuts[i], brodatz + "cropRow" + str(i) + ".png", 1)
    concatentationOfBrodatzTexturesIntoRows(dests, brodatz + outName, 0)

    # Destroy all sub crops (we can make this optional if we want!)
    deleteCroppedImages()

def createGridWithCircle(listOfBrodatzInts, circleInt, outName):

    listOfRowOutputs = []
    for i in range(len(listOfBrodatzInts)):
        brodatzCropInput = brodatz + "D" + str(listOfBrodatzInts[i]) + ".png"
        brodatzCropOutput = brodatz + "cropD" + str(listOfBrodatzInts[i]) + ".png"
        # 128x128 crops, in order to generate a 256x256 image
        cropTexture(256, 256, 384, 384, brodatzCropInput, brodatzCropOutput)
        listOfRowOutputs.append(brodatzCropOutput)
    subOuts = [listOfRowOutputs[x:x + 2] for x in xrange(0, len(listOfRowOutputs), 2)]
    dests = []
    for i in range(len(subOuts)):
        dest = brodatz + "cropRow" + str(i) + ".png"
        dests.append(dest)
        concatentationOfBrodatzTexturesIntoRows(subOuts[i], brodatz + "cropRow" + str(i) + ".png", 1)
    concatentationOfBrodatzTexturesIntoRows(dests, brodatz + "Nat5crop.png", 0)

    size = (128, 128)
    mask = Image.new('L', size, color=255)
    draw = ImageDraw.Draw(mask)
    draw.ellipse((0, 0) + size, fill=0)
    im = Image.open(brodatz + "D" + str(circleInt) + ".png")
    output = ImageOps.fit(im, mask.size, centering=(0.5, 0.5))
    output.paste(0, mask=mask)
    output.save(brodatz + 'circlecrop.png', transparency=0)

    img = Image.open(brodatz + 'circlecrop.png').convert("RGBA")
    img_w, img_h = img.size
    background = Image.open(brodatz + "Nat5crop.png")
    bg_w, bg_h = background.size
    offset = ((bg_w - img_w) / 2, (bg_h - img_h) / 2)
    background.paste(output, offset, img)
    background.save(brodatz + outName, format="png")
    deleteCroppedImages()

def createTexturePair(pair, outName):
    pathsToTemp = [brodatz + "D" + str(pair[0]) + ".png", brodatz + "D" + str(pair[1]) + ".png"]
    cropTexture(256, 256, 384, 384, pathsToTemp[0], brodatz + "outcrop1.png")
    cropTexture(256, 256, 384, 384, pathsToTemp[1], brodatz + "outcrop2.png")
    cropsToConcat = [brodatz + "outcrop1.png", brodatz + "outcrop2.png"]
    concatentationOfBrodatzTexturesIntoRows(cropsToConcat, outName, 1)
    deleteCroppedImages()

#--------------------------------------------------------------------------
# Create test images
#--------------------------------------------------------------------------

# Note that I did not write this to have an exhaustive approach in mind,
# where I pair all of the textures to every other texture.  If I did so,
# I would have made it a little more efficient, instead I just decided to
# use the images that were in the papers already.


# # We can use any of the 112 images from the Brodatz album here
# nat16 = [29,12,17,55,32,5,84,68,77,24,9,4,3,33,51,54]
# howManyPerRow = 4
# outName = "Nat16.png"
# createGrid(nat16, outName, howManyPerRow)
#
# grid4 = [3,68,17,77]
# howManyPerRow = 2
# outName = "grid4.png"
# createGrid(grid4, outName, howManyPerRow)

# #the last int is the circle in the middle of the image!
# nat5 = [77,55,84,17]
# circleInt = 24
# outName = 'Nat5.png'
# createGridWithCircle(nat5, circleInt, outName)
#
# texturePairs = [[17,77],[3,68],[3,17],[55,68]]
# count = 0
# for pair in texturePairs:
#     outName = brodatz + "pair" + str(count) + ".png"
#     createTexturePair(pair, outName)
#     count += 1
