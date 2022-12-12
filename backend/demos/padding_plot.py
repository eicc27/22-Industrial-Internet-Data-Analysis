import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from sklearn.datasets import make_circles
import sys
import os
from functools import partial
sys.path.append(os.path.abspath('..'))
from preproc.padding import Padding

def make_lost(data: np.ndarray, nums=10):
    rows = data.shape[0]
    lucky = []
    for _ in range(nums):
        lucky.append(np.random.randint(0, rows))
    for i in range(rows):
        if i == lucky:
            rows[i][0] = np.nan
    return data, lucky


class PaddingPlot:
    def __init__(self, method: str, fname: str, limits: tuple[tuple[float, float]] = ((-1, 1), (-1, 1)), frames=180) -> None:
        self.fname = fname
        self.xmin, self.xmax = limits[0]
        self.ymin, self.ymax = limits[1]
        self.frames = frames
        self.method = method
        data, self.labels = make_circles(
                n_samples=200, noise=0.03, random_state=0, factor=0.6)
        self.data, self.lucky = make_lost(data)
        c1, c2 = self.split_classes()
        self.X1, self.Y1 = c1.T
        self.X2, self.Y2 = c2.T

        self.fig, self.ax = plt.subplots()

    def find_dots(self):
        pad_data = Padding(self.data, 0, self.method).run()
        pad_data = Padding(pad_data, 1, self.method).run()
        pad_data_c1 = []
        pad_data_c2 = []
        for l in self.lucky:
            if self.labels[l] == 0:
                pad_data_c1.append(pad_data[l])
            else:
                pad_data_c2.append(pad_data[l])
        return np.array(pad_data_c1), np.array(pad_data_c2)

    def split_classes(self):
        data_c1 = []
        data_c2 = []
        for i, label in enumerate(self.labels):
            if label == 0:
                data_c1.append(self.data[i])
            else:
                data_c2.append(self.data[i])
        data_c1 = np.array(data_c1)
        data_c2 = np.array(data_c2)
        return data_c1, data_c2

    def gen_linspace(self, lost: np.ndarray, found: np.ndarray, index=1):
        linspaces = []
        for i in range(lost.shape[0]):
            l = np.linspace([lost[i][0], lost[i][1]], [
                found[i][0], found[i][1]], num=self.frames)
            linspaces.append(l.reshape(-1, 1, 2))
        return np.concatenate(linspaces, axis=1)

    def gen_animation(self, frame: np.ndarray, index=1):
        if index == 1:
            self.dots1.set_data(frame.T)
            return self.dots1,
        else:
            self.dots2.set_data(frame.T)
            return self.dots2,

    def animate(self):
        lost_dots_c1 = []
        lost_dots_c2 = []
        for l in self.lucky:
            if self.labels[l] == 0:
                lost_dots_c1.append([0, self.data[l][1]])
            else:
                lost_dots_c2.append([0, self.data[l][1]])
        lost_dots_c1 = np.array(lost_dots_c1)
        lost_dots_c2 = np.array(lost_dots_c2)
        self.ax.plot(self.X1, self.Y1, 'ro', markersize=1)
        self.ax.plot(self.X2, self.Y2, 'bo', markersize=1)
        self.dots1, = self.ax.plot(lost_dots_c1.T[0], lost_dots_c1.T[1], 'ro', markersize=6)
        self.dots2, = self.ax.plot(lost_dots_c2.T[0], lost_dots_c2.T[1], 'bo', markersize=6)
        found_dots_c1, found_dots_c2 = self.find_dots()
        a = animation.FuncAnimation(self.fig, partial(
            self.gen_animation, index=1), frames=self.gen_linspace(lost_dots_c1, found_dots_c1))
        b = animation.FuncAnimation(self.fig, partial(
            self.gen_animation, index=2), frames=self.gen_linspace(lost_dots_c2, found_dots_c2))
        a.save(self.fname, 'imagemagick', fps=60, extra_anim=[b])

PaddingPlot('mean', 'mean.gif').animate()
PaddingPlot('median', 'median.gif').animate()
PaddingPlot('knnr', 'knn.gif').animate()
PaddingPlot('forestr', 'forest.gif').animate()
