import numpy as np

def signValue_str(x, showPlus=True)->str:
    if x < 0:
        return f"- {abs(x)}"
    else:
        return f"+ {abs(x)}" if showPlus else f"{abs(x)}"

class Line():
    pt1 = None
    pt2 = None

    def __init__(self, pt1, pt2):
        self.pt1 = pt1
        self.pt2 = pt2

        self.a, self.b, self.c = self.computeABC(pt1, pt2)

    @classmethod
    def computeABC(cls, pt1, pt2):
        a = pt2[1] - pt1[1]
        b = pt1[0] - pt2[0]
        c = -a * pt1[0] + -b * pt1[1]

        return a, b, c
    
    def x2y(self, x):
        if self.b == 0:
            return self.pt1[1]
        return (-self.a * x - self.c) / self.b

    def y2x(self, y):
        if self.a == 0:
            return self.pt1[0]
        return (-self.b * y - self.c) / self.a
    
    def percentage2point(self, perct):
        x1, y1 = self.pt1[0], self.pt1[1]
        x2, y2 = self.pt2[0], self.pt2[1]

        x = x1 + perct * (x2 - x1)
        y = y1 + perct * (y2 - y1)

        return np.array([x, y])
    
    def __str__(self):
        return f"{signValue_str(self.a, showPlus=False)}x {signValue_str(self.b)}y {signValue_str(self.c)} = 0"
