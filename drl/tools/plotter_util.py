# We could see everything in tensorboard if we wanted, but what is the fun in that?
# Here we create a single threaded dynamic plotter with matplotlib
# For an interprocess plotter implementation go here: https://github.com/ctmakro/stanford-osrl/blob/master/plotter.py

import matplotlib.pyplot as plot

colors = {'red': [1, 0, 0]}


class Plotter:

    def __init__(self,
                 num_lines=1,
                 color='blue',
                 x_label='x',
                 y_label='y',
                 title='custom plot',
                 smooth=False):

        self.x = []
        self.y = []
        self.num_lines = num_lines
        self.ys = [[] for i in range(num_lines)]

        self.colors = [[
            (i * i * 7.9 + i * 19 / 2.3 + 17 / 3.1) % 0.5 + 0.2,
            (i * i * 9.1 + i * 23 / 2.9 + 31 / 3.7) % 0.5 + 0.2,
            (i * i * 11.3 + i * 29 / 3.1 + 37 / 4.1) % 0.5 + 0.2,
        ] for i in range(num_lines)]

        self.smooth = smooth

        self.fig = plot.figure()
        self.axis = self.fig.add_subplot(1, 1, 1)
        self.x_label = x_label
        self.y_label = y_label
        self.title = title
        plot.show(block=False)

    def show(self):

        self.axis.clear()
        self.axis.grid(color='#f0f0f0', linestyle='solid', linewidth=1)

        #this is important if self.x changes during somewhere else
        x = self.x
        self.axis.set_xlabel(self.x_label)
        self.axis.set_ylabel(self.y_label)
        self.axis.set_title(self.title)

        for _ in range(len(self.ys)):
            y = self.ys[_]
            c = self.colors[_]
            self.axis.plot(x, y, color=tuple(c))

        # Putting if outside is O(2n) but putting inside is O(n) more ifs
        if self.smooth:
            self.plot_smooth()

        self.fig.canvas.draw()

    def add_points(self, x, y, *args):
        """ Adds points to our plotter, at least one x and one y, but can take multiple ys """
        assert len(args) + 1 == self.num_lines, "points without a line"
        self.x.append(x)
        args = list(args)
        args.insert(0, y)
        for _ in range(len(args)):
            self.ys[_].append(args[_])

    def plot_smooth(self, alpha=0.5):
        x = self.x
        for _ in range(len(self.ys)):
            y = self.ys[_]
            c = self.colors[_]
            init = 5
            if len(y) > init:
                ysmooth = [sum(y[0:init]) / init] * init
                for i in range(init, len(y)):  # first order
                    ysmooth.append(ysmooth[-1] * 0.9 + y[i] * 0.1)
                for i in range(init, len(y)):  # second order
                    ysmooth[i] = ysmooth[i - 1] * 0.9 + ysmooth[i] * 0.1

                self.axis.plot(
                    x,
                    ysmooth,
                    lw=2,
                    color=tuple([cp**0.3 for cp in c]),
                    alpha=alpha)


if __name__ == '__main__':

    import time
    import ipdb
    pp = Plotter(num_lines=1)
    for _ in range(50):
        pp.add_points(_, _)
        pp.show()
        time.sleep(0.2)
    # pp.add_points(0, 1, 56)
    # pp.show()
    # time.sleep(1)
    # pp.add_points(1, 2, 57)
    # pp.show()
    # time.sleep(1)
    # pp.add_points(2, 67, 89)
    # pp.show()
    # time.sleep(1)
