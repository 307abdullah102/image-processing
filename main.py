import image_processing as ip
import os

def main():
    # read an image
    img = ip.readImage("breast-xray.png")
    # getting rows and columns
    """
    print(ip.getRow(img))
    print(ip.getCol(img))
    
    """
    # getting max and min pixel values
    """
    print(ip.getMax(img))
    print(ip.getMin(img))
    # getting pixel value for the given coordinates
    print(ip.getPix(img, 0, 0))

    """
    ## setting the pixel value for the given coordinates
    """
    nrows = ip.getRow(img)
    ncols = ip.getCol(img)
    settedImage = img
    for i in range(nrows):
        for j in range(ncols):
            settedImage = ip.setPix(settedImage, i, j, 0)  # set all pixels to 0
    ##
    # writing the array as an image to the folder
    dir = os.path.join(os.getcwd(), "output", "set-image")
    ip.createNewFolder(dir)
    ip.writeImage("./output/set-image/settedImage.png", settedImage)
    
    """
    # threshold function implementation
    """
    img = ip.readImage("test-pattern.png")
    dir = os.path.join(os.getcwd(), "output", "threshold-image")
    ip.createNewFolder(dir)
    ip.writeImage("./output/threshold-image/test-pattern.png", img)
    img = ip.thresholdImage(img, 128, 0, 255)
    ip.writeImage("./output/threshold-image/threshold-test-pattern.png", img)

    """
    # double image function implementation
    """
    img = ip.readImage("saltpep-board.png")
    dir = os.path.join(os.getcwd(), "output", "double-image")
    ip.createNewFolder(dir)
    ip.writeImage("./output/double-image/saltpep-board.png.png", img)
    img = ip.doubleImage(img)
    ip.writeImage("./output/double-image/doubled-saltpep-board.png", img)  
    
    """
    # half image function implementation
    """
    img = ip.readImage("saltpep-board.png")
    dir = os.path.join(os.getcwd(), "output", "half-image")
    ip.createNewFolder(dir)
    ip.writeImage("./output/half-image/saltpep-board.png", img)
    img = ip.halfImage(img)
    ip.writeImage("./output/half-image/half-of-saltpep-board.png", img)

    """
    # negative mask function implementation
    """
    img = ip.readImage("breast-xray.png")
    dir = os.path.join(os.getcwd(), "output", "negative-image")
    ip.createNewFolder(dir)
    ip.writeImage("./output/negative-image/breast-xray.png", img)
    img = ip.negativeMask(img)
    ip.writeImage("./output/negative-image/negative-breast-xray.png", img)
   
    """
    # logaritmic mask function implementation
    """
    img = ip.readImage("no-log-dft.png")
    dir = os.path.join(os.getcwd(), "output", "logaritmic-image")
    ip.createNewFolder(dir)
    ip.writeImage("./output/logaritmic-image/no-log-dft.png", img)
    img = ip.logaritmicMask(img)
    ip.writeImage("./output/logaritmic-image/logaritmic-no-log-dft.png", img)

    """
    # gamma mask function implementation
    """
    img  = ip.readImage("fractured-spine.png")
    img2 = ip.readImage("washed_out_aerial_image.png")
    dir = os.path.join(os.getcwd(), "output", "gamma-image")
    ip.createNewFolder(dir)
    ip.writeImage("./output/gamma-image/fractured-spine.png", img)
    img = ip.gammaMask(img, gamma = 0.6)
    ip.writeImage("./output/gamma-image/gamma-0_6.png", img)
    img = ip.gammaMask(img, gamma = 0.4)
    ip.writeImage("./output/gamma-image/gamma-0_4.png", img)
    img = ip.gammaMask(img, gamma = 0.3)
    ip.writeImage("./output/gamma-image/gamma-0_3.png", img)

    ip.writeImage("./output/gamma-image/washed_out_aerial_image.png", img2)
    img2 = ip.gammaMask(img2, gamma = 3)
    ip.writeImage("./output/gamma-image/gamma-3.png", img2)
    img2 = ip.gammaMask(img2, gamma = 4)
    ip.writeImage("./output/gamma-image/gamma-4.png", img2)
    img2 = ip.gammaMask(img2, gamma = 5)
    ip.writeImage("./output/gamma-image/gamma-5.png", img2)

    """
    # average filter mask implementation
    """
    img = ip.readImage("test-pattern.png")
    dir = os.path.join(os.getcwd(), "output", "average-filter-masked-image")
    ip.createNewFolder(dir)
    ip.writeImage("./output/average-filter-masked-image/test-pattern.png", img)
    img = ip.averageFilterMask(img, 3)
    ip.writeImage("./output/average-filter-masked-image/average-filter-3x3-image-test-pattern.png", img)
    img = ip.averageFilterMask(img, 5)
    ip.writeImage("./output/average-filter-masked-image/average-filter-5x5-image-test-pattern.png", img)

    """
    # median filter mask implementation
    """
    img = ip.readImage("saltpep-board.png")
    dir = os.path.join(os.getcwd(), "output", "median-filter-masked-image")
    ip.createNewFolder(dir)
    ip.writeImage("./output/median-filter-masked-image/saltpep-board.png", img)
    img = ip.medianFilterMask(img, 3)
    ip.writeImage("./output/median-filter-masked-image/median-filter-3x3-image-saltpep-board.png", img)
    img = ip.medianFilterMask(img, 5)
    ip.writeImage("./output/median-filter-masked-image/median-filter-5x5-image-saltpep-board.png", img)
    
    """
   # histogram equalization implementation
    """
    img1 = ip.readImage("coffee_bean_1.png")
    img2 = ip.readImage("coffee_bean_2.png")
    img3 = ip.readImage("coffee_bean_3.png")
    img4 = ip.readImage("coffee_bean_4.png")

    dir = os.path.join(os.getcwd(), "output", "histogram-equalization-image")
    ip.createNewFolder(dir)

    ip.writeImage("./output/histogram-equalization-image/coffee_bean_1.png", img1)
    ip.writeImage("./output/histogram-equalization-image/coffee_bean_2.png", img2)
    ip.writeImage("./output/histogram-equalization-image/coffee_bean_3.png", img3)
    ip.writeImage("./output/histogram-equalization-image/coffee_bean_4.png", img4)

    img1 = ip.histogramEqualization(img1)
    ip.writeImage("./output/histogram-equalization-image/hist-equalization-coffee_bean_1.png", img1)
    img2 = ip.histogramEqualization(img2)
    ip.writeImage("./output/histogram-equalization-image/hist-equalization-coffee_bean_2.png", img2)
    img3 = ip.histogramEqualization(img3)
    ip.writeImage("./output/histogram-equalization-image/hist-equalization-coffee_bean_3.png", img3)
    img4 = ip.histogramEqualization(img4)
    ip.writeImage("./output/histogram-equalization-image/hist-equalization-coffee_bean_4.png", img4)

    """
    # sharpening mask implementation
    """
    img = ip.readImage("blurry_moon.png")
    dir = os.path.join(os.getcwd(), "output", "sharpening-mask-image")
    ip.createNewFolder(dir)
    ip.writeImage("./output/sharpening-mask-image/blurry_moon.png", img)
    filteredImg = ip.sharpeningMask(img)
    #SCALING
    ip.writeImage("./output/sharpening-mask-image/filtered-blurry_moon.png", filteredImg)
    sharpeningImg = ip.subtract(img, filteredImg, False)
    ip.writeImage("./output/sharpening-mask-image/sharpening-blurry_moon.png", sharpeningImg)

    ip.writeImage("./output/sharpening-mask-image/blurry_moon.png", img)
    laplacedImg = ip.sharpeningLaplace(img)
    #SCALING
    ip.writeImage("./output/sharpening-mask-image/laplaced-blurry_moon.png", laplacedImg)
    sharpening_LaplacedImg = ip.subtract(img, laplacedImg, False)
    ip.writeImage("./output/sharpening-mask-image/sharpening-laplaced-blurry_moon.png", sharpening_LaplacedImg)

    """
    # sharping mask using median filter
    """
    img = ip.readImage("dipxe_text.png")
    dir = os.path.join(os.getcwd(), "output", "sharpening-mask-image-using-median-filter")
    ip.createNewFolder(dir)
    ip.writeImage("./output/sharpening-mask-image-using-median-filter/dipxe_text.png", img)
    medianImg = ip.medianFilterMask(img, 9)
    sharpeningImg = medianImg
    ip.writeImage("./output/sharpening-mask-image-using-median-filter/sharpening-dipxe_text.png", sharpeningImg)
    """

if __name__ == "__main__":
    main()

