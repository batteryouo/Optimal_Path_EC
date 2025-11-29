import cv2

import core
import map_generator

def main():
    mapGenerator = map_generator.Generator()
    mapGenerator.createFramework((500, 500), 10)
    mapGenerator.generate(20, 10)
    core.run()
    cv2.imshow("canvas", mapGenerator.canvas)
    cv2.waitKey(0)
if __name__ == "__main__":
    main()