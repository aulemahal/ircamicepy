"""
ircam_reader.py - Provides a function and a CLI to read the ascii output files of a IRBIS sequence.

The function can be called through the command line as:
> python ircam_reader.py -p prefix -f npy folder outputfile
Or for more info:
> python ircam_reader.py -h

It can also be imported and call in a python script.
>>> from ircam_reader import read_ir_asc

Written by Pascal Bourgault, 2018
Somewhere in the Arctic
"""
import numpy as np
import datetime as dt
import argparse
import os
from pickle import dump


def read_ir_asc(folder, prefix='irdata', extension='asc', verbose=False, sep='\t'):
	"""
	Reads all the IRBIS sequence files (as exported in ASCII) in a folder.
	
	Args:
		folder  - folder where to read the files. The files are read in alphanumerical order.
	
	Kwargs:
		prefix  - Prefix of the appropriate files in folder. 
				  Default: "irdata"
		extension. - Extension of the appropritate files in folder.
		          Default: "asc"
		verbose - Flag for the verbosity level of the function (for cli and script purposes)
	
	Returns:
		data    - The raw IR temperature data (3D numpy array)
		settings - The camera settings for this sequence
		params  - The parameters (list of dicts) of each frame.
		rois    - The stats for the exported regions of interest (3D numpy rec array)
	"""
	# find all corresponding files in folder and get their number. 
	allfiles = os.listdir(folder)
	filelist = [f for f in allfiles if f.startswith(prefix) and f.endswith(extension)]

	settings = {}
	nROIs = 0
	measline = 0
	# Read the first file to get the parameters, the size of the data and the ROIs
	with open(os.path.join(folder, filelist[0])) as f:
		for i, line in enumerate(map(lambda x: x.strip(), f)):
			if line == '[Settings]':
				sec = 'settings'
				continue
			elif line == '[MeasDefs]':
				sec = 'meas'
				measline = i + 2
			elif line == '[Data]':
				dataline = i
				break
			elif line is '':
				if sec == 'meas':
					nROIs = i - measline
				sec = None

			if sec == 'settings':
				if verbose: print(line)
				key, val = line.split('=')
				if ';' in val:
					val = val.split(';')
				settings[key] = parsestr(val)	
			elif sec == 'meas' and line.startswith('ID'):
				roi_fields = line.split('=')[1].split('\t')

	if verbose and nROIs: print('There are {:2d} ROI measurements with the following statistics: {}'.format(nROIs, roi_fields))
	# Create all other arrays and cycle through the files to fill them.
	# A recarray is an array where each element is like a dict. It has classic bracket and attribute access : rois[0, 1].Max is the max value of region 1 in image 0.
	rois = None if not nROIs else np.recarray((len(filelist), nROIs), dtype=[(field, float) for field in roi_fields])
	data = np.empty((len(filelist), settings['ImageWidth'], settings['ImageHeight']))
	params = []
	for j, file in enumerate(filelist):
		if verbose:
			print('Reading file {} (of {:3d}, {:.0%})'.format(file, len(filelist), float(j) / len(filelist)))
		with open(os.path.join(folder, file)) as f:
			for i, line in enumerate(map(lambda x: x.strip(), f)):
				if i >= measline and i < measline + nROIs:
					rois[j, i - measline] = tuple(parsestr(line.split('=')[1].split('\t')))
				elif i > dataline:
					data[j, :, -(i - dataline - 1)] = [float(x.replace(',', '.')) for x in line.split(sep)[:data.shape[1]]] # Negative to reverse the image (origin is now in the lower left corner)
				elif line == '[Parameter]':
					sec = 'params'
					params.append(dict())
				elif line == '':
					sec = None
				elif sec == 'params':
					key, val = line.split('=')
					params[j][key] = val

	if params:
		for framepar in params:
			d, mo, y = framepar['RecDate'].split('.')
			h, mi, s = framepar['RecTime'].split(':')
			framepar['RecDateTime'] = dt.datetime(int(y), int(mo), int(d), int(h), int(mi), int(s))
	data = data.swapaxes(1, 2)  # Transpose to have T, Y, X
	if verbose: print('\nDone.')
	return data, settings, params, rois


def parsestr(s):
	"""
	Simple function to transform a string to an appropriate python object.
	Cycles through lists. Supports floats ( . or , ), ints and strings (default)
	"""
	if isinstance(s, list):
		return [parsestr(ss) for ss in s]
	try:
		n = float(s.replace(',', '.'))
		if n == int(n):
			return int(n)
		return n
	except ValueError:
		return s


reader_parser = argparse.ArgumentParser(prog='IR ASC Reader', description='Reads sequence data from IRCam exported ASCII files.')
reader_parser.add_argument('--prefix', '-p', help='Prefix of the files in folder. [irdata]', default='irdata', type=str)
reader_parser.add_argument('--sublen', '-l', help='Number of file per sub-sequences. Put 0 if single sequence. [0]', default=0, type=int)
reader_parser.add_argument('--format', '-f', help='Output format. pic saves to pickled dict, npz saves all to a npz, npy saves three files (params in json + two npy arrays), mat saves all to a matlab file but needs scipy and nc to a netCDF [pic]',
				   		   default='pic', choices=['pic', 'npy', 'npz', 'mat', 'nc'], type=str)
reader_parser.add_argument('--verbose', '-v', help='Verbose flag', action='store_true', default=False)
reader_parser.add_argument('folder', help='Folder from where to read the files.', type=str)
reader_parser.add_argument('output', help='File where to save the arrays (without extension).', type=str)
# This line assures that the CLI script only starts when the file is called directly.
if __name__ == '__main__':
	args = reader_parser.parse_args()

	print('Reading files from {}'.format(args.folder))
	data, settings, params, rois = read_ir_asc(args.folder, prefix=args.prefix, sublen=args.sublen, verbose=args.verbose)

	if args.format == 'pic':
		with open(args.output + '.pic',  'wb') as f:
			dump(dict(data=data, rois=rois, params=params, **settings), f)
	elif args.format == 'npy':
		np.save(args.output + '_data.npy', data)
		if rois is not None: np.save(args.output + '_rois.npy', rois)
		import json
#		with open(args.output + '_params.txt', 'w') as f:
#			json.dump(f, params)
		print('Not saving the parameters of each frame, stupid json.')
		with open(args.output + '_settings.txt', 'w') as f:
			json.dump(f, settings)
	elif args.format == 'npz':
		np.savez(args.output, data=data, rois=rois, params=params, settings=settings)
	elif args.format == 'mat':
		import scipy.io as sio
		sio.savemat(args.output, mdict=dict(data=data, rois=rois, params=params, settings=settings))
	elif args.format == 'nc':
		from netCDF4 import Dataset
		d = Dataset(args.output + '.nc', 'w')
		d.createDimension('X', size=settings['ImageWidth'])
		d.createDimension('Y', size=settings['ImageHeight'])
		d.createDimension('Frame', size=None)
		v = d.createVariable('Temp', dimensions=('Frame', 'Y', 'X'))
		v.setncattr('Units', 'degC')
		v.setncattr('Description', 'Temperature data from the IR Cam')
		v[:] = data
		for field in rois.dtype.fields.keys():
			for iroi in range(rois.shape[1]):
				v = d.createVariable('R{}_{}'.format(iroi, field), dimensions=('Frame',))
				v.setncattr('Units', 'degC')
				v[:] = rois[:, iroi][field]
		for k, v in params.items():
			d.setncattr(k, v)
		
	print('{} files read and saved in {}.'.format(data.shape[0], args.output + '.' + args.format))

