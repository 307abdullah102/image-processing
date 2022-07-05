import imageio as iio
import numpy as np
import math
#ignore warning
np.seterr(over='ignore')
#print all numpy array
#np.set_printoptions(threshold=sys.maxsize)

def readImage(fname):
    img = iio.v3.imread(fname)
    return img

def writeImage(fname, img):
    iio.v3.imwrite(fname, img)

def getRow(img):
    return img.shape[0]

def getCol(img):
    return img.shape[1]

def getMax(img):
    max = -10000
    nRows = getRow(img)
    nCols = getCol(img)
    for i in range(nRows):
        for j in range(nCols):
            if max < img[i, j]:
                max = img[i, j]
    return max

def getMin(img):
    min = 10000
    nRows = getRow(img)
    nCols = getCol(img)
    for i in range(nRows):
        for j in range(nCols):
            if min > img[i, j]:
                min = img[i, j]
    return min

def getPix(img, rows, cols):
    return img[rows, cols]

def setPix(img, rows, cols, value):
    tempImg = img.copy()
    tempImg[rows, cols] = value
    return tempImg

def thresholdImage(img, thVal, lowVal, highVal):
    tempImg = img.copy()
    nrows = getRow(tempImg)
    ncols = getCol(tempImg)
    for i in range(nrows):
        for j in range(ncols):
            if tempImg[i, j] <= thVal:
                tempImg[i, j] = lowVal
            else:
                tempImg[i, j] = highVal
    return tempImg

def doubleImage(img):
    nrows = getRow(img)
    ncols = getCol(img)
    doubledImage = np.zeros([2*nrows, 2*ncols], dtype = np.uint8)

    for i in range(nrows):
        for j in range(ncols):
            doubledImage[2*i, 2*j] = img[i, j]
    # when we doubled image and equalize pixels, we just filled even indexes
    # the following two for loops are for fill the odd columns
    for i in range(2*nrows):
        for j in range(0,(2*ncols-2),2):
            doubledImage[i, j+1] = np.uint8((float(doubledImage[i,j]) + float(doubledImage[i,j+2]))/2)

    # the following two for loops are for fill the odd rows
    for j in range(2*ncols):
        for i in range(0,(2*nrows-2),2):
            doubledImage[i+1, j] = np.uint8((float(doubledImage[i, j]) + float(doubledImage[i+2, j]))/2)

    return doubledImage

def halfImage(img):
    nrows = getRow(img)
    ncols = getCol(img)
    hnrows = np.uint8(nrows / 2) - 1
    hncols = np.uint8(ncols / 2) - 1
    halfOfImage= np.zeros([int(nrows/2), int(ncols/2)], dtype = np.uint8)

    for i in range(hnrows):
        for j in range(hncols):
            halfOfImage[i, j] = img[2*i, 2*j]
    return halfOfImage

def negativeMask(img):
    tempImage = img.copy()
    nrows = getRow(img)
    ncols = getCol(img)
    maxPix = getMax(img)
    for i in range(nrows):
        for j in range(ncols):
            tempImage[i, j] = maxPix - tempImage[i, j]
    return tempImage

def logaritmicMask(img):
    tempImage = img.copy()
    nrows = getRow(img)
    ncols = getCol(img)
    maxPix = getMax(img)
    const = 1
    maxOfLog =  const * np.log(1 + (maxPix - 1))
    scaleConst = maxPix/maxOfLog

    for i in range(nrows):
        for j in range(ncols):
            tempImage[i, j] = scaleConst * const * np.log(1 + tempImage[i, j])
    return tempImage

def gammaMask(img, gamma):
    tempImage = img.copy()
    nrows = getRow(img)
    ncols = getCol(img)
    maxPix = getMax(img)
    const = 1
    maxOfGamma = const * math.pow(maxPix, gamma)
    scaleConst = maxPix/maxOfGamma

    for i in range(nrows):
        for j in range(ncols):
            tempImage[i, j] = scaleConst * const * math.pow(tempImage[i, j], gamma)
    return tempImage

def averageFilterMask(img, sizeOfFilter):
    tempImage = img.copy()
    nrows = getRow(img)
    ncols = getCol(img)
    pixSumValue = 0

    for i in range(nrows):
        for j in range(ncols):
            # sizeOfFilter = 3 için maskede merkez pikselin soluna ve sagina
            # 1 birim gidip ilgili pikseller toplanmalıdır. (-(3-1)/2, (3-1)/2 + 1) -> (-1, 2, 1)
            # bu da -1 0 1 sol, merkez ve saga gitmek anlamına gelir.
            # Kodun 5x5, 7x7 ... için de gecerli oldugunu kagit uzerinde deneyebilirsiniz.
            for filtRows in range(int(-(sizeOfFilter - 1)/2), int((sizeOfFilter - 1)/2 + 1), 1):
                for filtCols in range(int(-(sizeOfFilter - 1)/2), int((sizeOfFilter - 1)/2 + 1), 1):
                    if (filtRows + i) >= 0 \
                            and (filtRows + i) < nrows \
                            and (filtCols + j) >= 0 \
                            and (filtCols + j) < ncols:
                        pixSumValue += tempImage[(i + filtRows), (j + filtCols)]
                    else:
                        pixSumValue += 0
            tempImage[i, j] = pixSumValue / (math.pow(sizeOfFilter, 2))
            pixSumValue = 0
    return tempImage

def medianFilterMask(img, sizeOfFilter):
    tempImage = img.copy()
    nrows = getRow(img)
    ncols = getCol(img)
    medianArray = np.zeros(sizeOfFilter * sizeOfFilter)
    counter = 0
    for i in range(nrows):
        for j in range(ncols):
            # sizeOfFilter = 3 için maskede merkez pikselin soluna ve sagina
            # 1 birim gidip ilgili pikseller toplanmalıdır. (-(3-1)/2, (3-1)/2 + 1) -> (-1, 2, 1)
            # bu da -1 0 1 sol, merkez ve saga gitmek anlamına gelir.
            # Kodun 5x5, 7x7 ... için de gecerli oldugunu kagit uzerinde deneyebilirsiniz.
            for filtRows in range(int(-(sizeOfFilter - 1)/2), int((sizeOfFilter - 1)/2 + 1), 1):
                for filtCols in range(int(-(sizeOfFilter - 1)/2), int((sizeOfFilter - 1)/2 + 1), 1):
                    if (filtRows + i) >= 0 \
                            and (filtRows + i) < nrows \
                            and (filtCols + j) >= 0 \
                            and (filtCols + j) < ncols:
                        medianArray[counter] = tempImage[(i + filtRows), (j + filtCols)]
                    else:
                        medianArray[counter] = 0
                    counter += 1
            medianArray = sortArray(medianArray, sizeOfFilter)
            tempImage[i, j] = medianArray[int((sizeOfFilter * sizeOfFilter - 1)/2)]
            counter = 0
    return tempImage

def sortArray(medianArray, sizeOfFilter):
    size = sizeOfFilter * sizeOfFilter
    tempArr = medianArray.copy()
    for i in range(size):
        min = i
        for j in range(i+1, size, 1):
            if tempArr[j] < tempArr[min]:
                min = j
                temp = tempArr[i]
                tempArr[i] = tempArr[min]
                tempArr[min] = temp

    return tempArr