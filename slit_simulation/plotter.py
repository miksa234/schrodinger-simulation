#!/usr/bin/python3.9

import numpy as np
import matplotlib.pyplot as plt
import os


def plotter(U, V, dt, xx, yy, name:str):

    n = len(xx)
    path = './.plotcache'
    if os.path.exists(path) != True:
        os.mkdir(path)

    print('Making gif...')
    for i in range(0, len(U)):
        levels = np.linspace(0, max(U[i].reshape(n*n)), 200)
        fig, ax = plt.subplots(figsize=[7,5])

        c = ax.contourf(xx, yy, U[i], levels=levels, zorder=1,\
                        cmap=plt.cm.inferno)

        ax.contour(xx, yy, V.reshape(n, n), extend='both',\
                   cmap=plt.cm.binary)

        ax.set_title(f'Time {round(i*dt, 2)}')
        fig.colorbar(c)

        plt.savefig(f'{path}/img-{i}.png')
        plt.close()

    plt.plot(U[-1][15])
    plt.savefig('./inteferenz.png')
    plt.close()

    os.system(f'ffmpeg -start_number 0 -i {path}/img-%d.png {name}.gif')
    os.system(f'ffmpeg -start_number 0 -i {path}/img-%d.png {name}.mp4')
    os.system('rm -rf ' + path)
    os.system('rm -rf __pycache__')
