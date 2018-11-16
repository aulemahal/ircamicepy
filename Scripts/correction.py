"""
correction.py - Functions and script to correct the camera data.


Functions:

    generate_correction_fun : Generate a correction function that relates the needed correction factor with the incidence angle 

    correct_data : Correct a thermal image from a correction matrix
	
	The suggested use of this is as follows:
	1. Create an image that we can suppose uniform and taken in the same conditions as the rest of the ice dataset. 
		For example, it can be one (or more averaged) image where only open water can be seen.
	2. Extract the supposed uniform temperature. For example: the water temperature.
		In our example, it would be the temperature a the bottom of the image, where the incidence angle and its effect are minimal.
	3 (if it applies): If parts of instruments, ship, etc are visible in the images, crop them out.
					   Also, create the angle array with angles_tcam and crop the same parts out.
	4. Call generate_correction_fun with the temperature image, the inclination angle and the theoritical reference temperature.
		The result is a poly1d instance.
	5. Call the poly1d with the angles matrix to get a correction matrix
	6. Correct any data taken in the same conditions with correct_data(temp, corr).
	
CLI interface:
	This part is meant to be used on sequences of open water (uniform T).
    When lauched as:
    > python correction.py 
    this file provides a cli utility computing the correction matrix for a given angle and using a sequence of uniform images.
    It takes the mean of the usable area and passes it to the generate_correction_fun function. It then computes and saves the
    correction matrix to file.
    Use "-h" to get more info about the usable arguments.
    
Written by Pascal Bourgault, 2018
Somewhere in the Arctic
"""
import argparse
from angle_correct import angles_tcam
from  ircam_reader import reader_parser, read_ir_asc
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt


def generate_correction_fun(temp, angles, Tref, fov=10, order=3, return_ratio=False):
    """Generate a correction function for IRCam images from a supposedly uniform image.

    Args:
    temp   - 2D thermal image (deg C) that we suppose to be uniformely at the Tref temperature, the only error coming from the different incidence angles
    angles - Either the angle from the horizontal of the camera or an array with the same shape as temp with the incidence angle of each pixel
    Tref   - The supposed uniform temperature

    Kwargs:
    fov    - If angles is a scalar, the camera's fov (def : 10 mm)
    order  - The order of the fitted polynomial (def : 3)
    return_ratio  - Whether to return the error ratio of temperatures

    The correction is found by looking at the ratio of Tref / T_measured as a function of the incidence angle.
    A polynomial function is fitted and can be used to find a correction matrix when applied to the angle matrix.

    Returns:
        - a numpy.poly1d object of the fitted polynomial
    """
    if np.isscalar(angles):
        angles = angles_tcam(angles, fov=fov).T

    T_ratio = (Tref + 273.15) / (temp + 273.15)
    
    fitfun = np.poly1d(np.polyfit(angles.ravel(), T_ratio.ravel(), order))

    if return_ratio:
        return fitfun, T_ratio

    return fitfun


def correct_data(temp, corr):
    """Correct a thermal array in deg C from a correction matrix."""
    return (temp + 273.15) * corr - 273.15


if __name__ == '__main__':

    def crop_arg_parser(argstr):
        inds = [int(s) if s else None for s in argstr.split(',')]
        return [slice(inds[0], inds[1]), slice(inds[2], inds[3])]

    def T0_arg_parser(argstr):
        try:
            return float(argstr)
        except ValueError:
            return crop_arg_parser(argstr)

    parser = argparse.ArgumentParser(parents=(reader_parser,), add_help=False, prog='IR Corr Gen', description='Correction files generation for IRCam data. Shows some nice plots with verbose on. NetCDF not possible.')
    parser.add_argument('--crop', help='Part of the image to crop out from the computations (x0, x1, y0, y1) [,,,]', type=crop_arg_parser, default=',,,')
    parser.add_argument('-T', '--ref', help='Reference temperature to take when computing the ratio (deg C). Can be given as an area of which the mean is computed (same format as crop). [,50,200,440]', type=T0_arg_parser, default=',50,200,440')
    parser.add_argument('--savefigs', help='Save stats and figures to file. Needs verbose.', action='store_true', default=False)
    parser.add_argument('-a', '--angle', help='Camera\'s angle form the horizontal [30]', default=30, type=float)
    parser.add_argument('--order', help='Order of the polynomial used to fit the error vs angle relation [3]', type=int, default=3)
    parser.add_argument('--fov', help='FOV of the objective (mm) [10]', default=10)
    args = parser.parse_args()

    temp, rois, params = read_ir_asc(args.folder, prefix=args.prefix, verbose=args.verbose)

    if args.verbose: print('Finished reading files. Computing reference temperature and correction.')

    if isinstance(args.ref, list):
        Tref = temp[:, args.crop[1], args.crop[0]][:, args.ref[1], args.ref[0]].mean()
    else:
        Tref = args.ref

    angles = angles_tcam(args.angle, fov=args.fov).T

    fitfun, T_ratio = generate_correction_fun(temp.mean(axis=0)[args.crop[1], args.crop[0]], angles[args.crop[1], args.crop[0]], Tref, order=args.order, return_ratio=True)

    corr = fitfun(angles)

    if args.verbose: print('Saving correction matrix.')
    
    if args.format == 'pic':
        with open(args.output + '.pic',  'wb') as f:
            dump(corr, f)
    elif args.format == 'npy' or args.format == 'npz':
        np.save(args.output + '.npy', corr)
    elif args.format == 'mat':
        import scipy.io as sio
        sio.savemat(args.output, mdict=dict(correction=corr))

    if args.verbose:
        print('Plotting results.')
        plt.style.use('ggplot')
        figR, axR = plt.subplots()
        axR.scatter(angles, T_ratio, label='All pixels')
        angs = np.linspace(angles.min(), angles.max())
        axR.plot(angs, fitfun(angs), color='b', label='Fit used for the correction')
        axR.set_title('Ratio of reference to measured temperature as a function of incidence angle.')
        axR.set_ylabel('Ratio of reference to measured temperature')
        axR.set_xlabel('Incidence angle [rad]')
        axR.legend()

        figC, axC = plt.subplots()
        imC = axC.imshow(correct_data(temp[0], corr), origin='lower')
        cb = figC.colorbar(imC)
        cb.set_label('Temperature [$^\circ$C]')
        axC.set_title('Corrected first frame')

        if args.savefigs:
            figR.savefig(args.output + '_ratios.png')
            figC.savefig(args.output + '_corrected.png')
        
        plt.show()

