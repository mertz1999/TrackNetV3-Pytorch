import numpy as np
import matplotlib.pyplot as plt

def genHeatMap(w, h, cx, cy, r, mag):
    """

    generate heat map of tracking badminton
    param:
    w: width of output heat map 
    h: height of output heat map
    cx: x coordinate of badminton
    cy: y coordinate of badminton
    r: radius of circle generated
    mag: factor to change range of grayscale

    """
    if cx == 0 and cy == 0:
        return np.zeros((h, w))

    x, y = np.meshgrid(np.linspace(1, w, w), np.linspace(1, h, h))
    heatmap = ((y - (cy + 1))**2) + ((x - (cx + 1))**2)
    heatmap[heatmap <= r**2] = 1
    heatmap[heatmap > r**2] = 0
    return heatmap*mag


# heat_map = genHeatMap(1280, 720, 500, 200, 5, 1)

# plt.imshow(heat_map);plt.show()
# print(heat_map)