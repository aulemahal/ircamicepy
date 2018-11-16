"""
utils.py - Some utilities to play with the IR camera data.

Written by Pascal Bourgault, 2018
Somewhere in the Arctic
"""
import numpy as np
import pandas as pd
import datetime as dt
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.animation as anim
from angle_correct import project_image


def plot_movie(data, dt=1, cam_incl_angl=None, height=None, figkwa=None, anikwa=None, **imkwa):
    """Plot a matplotlib animation from a 3D array.

    Args and kwargs:
    data   - The numpy nd array, as (time, X, Y)
    cam_incl_angl - The camera's inclination from the horizontal. Def: None
                    If given, makes an animation from the re-projected images and height must be given.
    height - If cam_incl_angl is given, the height at which the camera is mounted (m), passed to project_image()
    dt     - Either the timestep in seconds between two frames, or a vector of stringable-values as timestamps.
    figkwa - Dict of figure-related keyword arguments, passed to the plt.subplots() call
    anikwa - Dict of animation-related keyword arguments, passed to the matplotlib.animation.FuncAnimation() call.
    
    All other kwargs are passed to the pyplot.imshow() (or the pyplot.pseudocolormesh()) calls. For example : cmap, vmin, vmax
    Creates a plot with a colorbar corresponding to data[0],
       + a title as "Brightness temp. | {timestamp}"

    Returns:
        - the figure handle
        - the ax handle
        - the animation handle
    """

    figkwa = {} if figkwa is None else figkwa
    anikwa = {} if anikwa is None else anikwa
    fig, ax = plt.subplots(**figkwa)
    
    if cam_incl_angl is None:
        imkwa.setdefault('origin', 'lower')
        im = ax.imshow(data[0], **imkwa)
    else:
        X, Y = project_image(cam_incl_angl, height)
        im = ax.pcolormesh(X, Y, data[0], **imkwa)
        ax.set_aspect('equal')
    
    t = ax.set_title('Brightness temperature')

    cb = fig.colorbar(im)
    cb.set_label('Temperature [$^\circ$C]')

    if np.isscalar(dt):
        units = 's'
        if dt > 120:
            dt /= 60
            units = 'mn'
        times = ['{:.2f} {}'.format(i * dt, units) for i in range(data.shape[0])]
    else:
        times = ['{}'.format(time) for time in dt]
    
    def animate(i, im=None):
        if cam_incl_angl is None:
            im.set_data(data[i])
        else:
            ax.get_children()[0].remove()
            im = ax.pcolormesh(X, Y, data[i], **imkwa) 
        t.set_text('Brightness temp. | {}'.format(times[i]))
        return im, t, ax

    ani = anim.FuncAnimation(fig, animate, frames=data.shape[0], fargs=(im,), **anikwa)
    return fig, ax, ani


def read_dship(filename, short_names=True, sep='\t', infer_datetime_format=True,
               parse_dates=True, index_col=0, header=[0, 2], **kwargs):
    """Convenience function to pandas.read_csv() with presetted defaults to read from Dship.
    
    Reads tab-separated values, first column as a datetime index and three row header : names, spot, units.
    If short_names is True (default), drops the long name of all columns and the units to keep only the last part.
    """
    data = pd.read_csv(filename, sep=sep, infer_datetime_format=infer_datetime_format, 
                       parse_dates=parse_dates, index_col=index_col, header=header, **kwargs)
    if short_names:
        data.columns = pd.Index([name.split('.')[-1] for name in data.columns.get_level_values(0)])
    return data


def read_pinocchio(filename):
    def date_parser(argstr):
        return dt.datetime(*[int(num) for num in argstr.split()])
    
    return pd.read_csv(filename, sep='\t', parse_dates=True, date_parser=date_parser,
                       index_col=0, skiprows=1, names=['coverage', 'temperature', 'inhomogeinity'])
