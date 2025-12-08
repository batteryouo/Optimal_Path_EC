import cv2

import core
import map_generator

def main():
    mapGenerator = map_generator.Generator(333)
    mapGenerator.createFramework((1000, 1000), 10)
    pts = None
    img = None
    while pts is None:
        pts, img = mapGenerator.generate(20, 100, 900, 30, 30)
    color_map = mapGenerator.drawLineAndPoints(img, pts)
    cv2.namedWindow("canvas", 0)
    cv2.imshow("canvas", color_map)
    
    core.ec.PathIndividual(img, pts)
    cv2.waitKey(0)

if __name__ == "__main__":
    main()