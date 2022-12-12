import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from sklearn.datasets import make_circles
import sys
import os
from functools import partial
sys.path.append(os.path.abspath('..'))
from preproc.normalization import Norm


class NormPlot:
    def __init__(self, method: str, fname: str, limits: tuple[tuple[float, float]] = ((-1, 1), (-1, 1)), frames=180) -> None:
        self.fname = fname
        self.xmin, self.xmax = limits[0]
        self.ymin, self.ymax = limits[1]
        self.frames = frames
        data, self.labels = make_circles(
                n_samples=500, noise=0.03, random_state=0, factor=0.6)

        c1, c2 = self.split_classes(data=data)
        self.X1, self.Y1 = c1.T
        self.X2, self.Y2 = c2.T

        norm_data = Norm(data, 0, method).run()
        norm_data = Norm(norm_data, 1, method).run()
        c1, c2 = self.split_classes(data=norm_data)
        self.x1, self.y1 = c1.T
        self.x2, self.y2 = c2.T

        self.fig, self.ax = plt.subplots()

    def split_classes(self, data: np.ndarray):
        data_c1 = []
        data_c2 = []
        for i, label in enumerate(self.labels):
            if label == 0:
                data_c1.append(data[i])
            else:
                data_c2.append(data[i])
        data_c1 = np.array(data_c1)
        data_c2 = np.array(data_c2)
        return data_c1, data_c2

    def gen_linspace(self, index=1):
        linspaces = []
        if index == 1:
            e = enumerate(self.X1)
        else:
            e = enumerate(self.X2)
        for i, _ in e:
            if index == 1:
                l = np.linspace([self.X1[i], self.Y1[i]], [
                    self.x1[i], self.y1[i]], num=self.frames)
            else:
                l = np.linspace([self.X2[i], self.Y2[i]], [
                    self.x2[i], self.y2[i]], num=self.frames)
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
        self.ax.set_xlim(self.xmin, self.xmax)
        self.ax.set_ylim(self.ymin, self.ymax)
        self.dots1, = self.ax.plot(self.X1, self.Y1, 'ro', markersize=1)
        self.dots2, = self.ax.plot(self.X2, self.Y2, 'bo', markersize=1)
        a = animation.FuncAnimation(self.fig, partial(
            self.gen_animation, index=1), frames=self.gen_linspace(1))
        b = animation.FuncAnimation(self.fig, partial(
            self.gen_animation, index=2), frames=self.gen_linspace(2))
        print(f'Saving {self.fname}...')
        a.save(self.fname, 'imagemagick', fps=60, extra_anim=[b])


NormPlot('zscore', 'zscore.gif', ((-3, 3), (-3, 3))).animate()
NormPlot('minmax', 'minmax.gif').animate()
NormPlot('sigmoid', 'sigmoid.gif').animate()