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
    img = mapGenerator.drawLineAndPoints(img, pts)
    core.run()
    cv2.imshow("canvas", img)
    cv2.waitKey(0)

if __name__ == "__main__":
    main()