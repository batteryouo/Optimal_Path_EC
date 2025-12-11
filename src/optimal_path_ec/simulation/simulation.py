import math

import cv2
import numpy as np

class Simulation():
    def __init__(self, model, states, theta_array, pathLine, map):
        self.model = model
        self.states = states
        self.theta_array = theta_array
        self.pathLine = pathLine
        self.map = map
    def run(self, v = 1):
        reach = False
        ind = 0
        dt = 0
        while not reach:
            if ind >= len(self.pathLine) - 1:
                ind = -1            
            local_target = self.pathLine[ind].percentage2point(self.states[ind][0])
            robot_point = self.pathLine[ind].percentage2point(dt)

            dt += 0.01
            if abs(dt - self.states[ind][0]) < 0.02:
                theta = self.theta_array[ind]
                w = self.model.calW(1, theta)
                toward = self.pathLine[ind].theta
                endOfTurn = self.model.calEndXY(robot_point, theta, self.pathLine[ind].theta, self.pathLine[ind+1].theta)
                while(np.linalg.norm(endOfTurn - robot_point) > 1):
                    robot_point = self.model.calXY(robot_point, toward, v, 0.1 , w)
                    toward = self.model.calToward(toward, w, 0.1)
                    robot_img = self.draw_robot(robot_point[1], robot_point[0], np.pi/2 - toward)
                    cv2.namedWindow("bot", 0)
                    cv2.imshow("bot", robot_img)
                    key = cv2.waitKey(1)
                    if key == 27:
                        return
                dt = self.states[ind][1]
                ind += 1
                
                
            robot_img = self.draw_robot(robot_point[1], robot_point[0], np.pi/2 -  self.pathLine[ind].theta)
            cv2.namedWindow("bot", 0)
            cv2.imshow("bot", robot_img)
            key = cv2.waitKey(1)
            if key == 27:
                return

    def show(self, v = 1):

        robot_img = self.draw_robot(self.pathLine[0].pt1[1], self.pathLine[0].pt1[0], self.pathLine[0].theta)

        cv2.namedWindow("map", 0)
        cv2.imshow("map", robot_img)
        cv2.waitKey(0)
    def draw_robot(self, x, y, angle_radians, radius=3, color=(0, 255, 0), thickness=2):

        center = (int(x), int(y))
        img = cv2.circle(self.map, center, radius, color, thickness)

        end_x = x + radius * math.cos(angle_radians)
        end_y = y + radius * math.sin(angle_radians)
        end_point = (int(end_x), int(end_y))

        return cv2.arrowedLine(img, center, end_point, (0, 0, 255), thickness, tipLength=1.0)
