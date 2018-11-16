#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 30 2018

@author: Dmitrii Murashkin
"""
from __future__ import print_function

import numpy as np
import cv2
from cv2 import medianBlur
from cv2 import bilateralFilter
from skimage.segmentation import watershed
from skimage.segmentation import mark_boundaries
from skimage.segmentation import find_boundaries
from skimage.filters import sobel
from skimage.morphology import skeletonize
from skimage.measure import regionprops
import networkx as nx

from IceThickness import calc_h_i as calculate_ice_thickness
from angle_correct import angles_tcam as angle_correction
from ircam_reader import read_ir_asc


def find_local_maxima(arr):
    result = np.zeros_like(arr, dtype=np.bool)
    for i in range(1, arr.shape[0] - 1):
        for j in range(1, arr.shape[1] - 1):
            if (arr[i-1, j] < arr[i, j] > arr[i+1, j]) and (arr[i, j-1] < arr[i, j] > arr[i, j+1]):
                result[i, j] = True
    return result


def segment_image(arr):
    edges = sobel(frame_norm)
    edges_bin = np.where(edges > 0.03, True, False)
    edges_bin = skeletonize(edges_bin)
    from skimage.morphology import remove_small_objects
    remove_small_objects(edges_bin, min_size=2, in_place=True)
    distance = cv2.distanceTransform((~edges_bin).astype(np.uint8), cv2.DIST_L2, 5)
    import scipy.ndimage as nd
    from skimage.morphology import disk
    
    maxima = np.where(distance == nd.maximum_filter(distance, size=(3, 3)), True, False)
    seed = nd.binary_dilation(maxima, structure=disk(1))
#    seed = np.where(distance > 10, True, False)
    ret, markers = cv2.connectedComponents(seed.astype(np.uint8))
    segm = watershed(-distance, markers)
    return segm
    

def merge_neighbouring_segments(img, props):
    """ Some ice floes are divided into several segments.
        Therefore, segments that have a common boundary and have similar median temperatures are merged.
    """
    dic = props.copy()
    dic[0] = +10
    boundaries = find_boundaries(img, mode='outer')
    mask = np.where(img > 0, True, False) * boundaries           # mask of common boundaries
    merge = np.where(mask, img, 0)
    merge_list = np.where(merge > 0)
    
    """ Let's create a graph of neighboring segments """
    G = nx.Graph()
    for x, y in zip(merge_list[0], merge_list[1]):
        for l in range(x - 1, x + 2):
            for m in range(y - 1, y + 2):
#                print(merge[x][y])
#                print(merge[l][m])
#                print(dic[merge[x][y]], dic[merge[l][m]])
                try:
                    if np.abs(dic[merge[x][y]] - dic[merge[l][m]]) <= 0.1:
                        G.add_edge(merge[x][y], merge[l][m])
                except:
                    continue
    try:
        G.remove_node(0)
    except nx.exception.NetworkXError:
        pass
    segments_to_combine = [list(component) for component in nx.connected_components(G) if len(component) > 1]
    
    'segments to combine'
    """ Now, join neighboring segments by assigning one segment id to all pixels of neighboring segments.
        the id of a new segment is a minimal id of a group of segments to merge
    """
    for segments in segments_to_combine:
        target_segment = min(segments)
        remove_segments = segments[:]
        remove_segments.remove(target_segment)
        for segment in remove_segments:
            img[img == segment] = target_segment
    return img
            
            

if __name__ == '__main__':
    pass
    data_src, _, _, _ = read_ir_asc('/home/dmitrii/polarstern/data/scientists/PS115_2/ArcTrain/Sea Ice/Infrared Camera/Data/sea_ice/180922_143758/ascii', prefix='firstice', verbose=True)
    camera_incl_angle = 30
    frame_number = 1
    frame = medianBlur(data_src[frame_number].astype(np.float32), 5)
#    frame = bilateralFilter(data_src[frame_number].astype(np.float32), 5, 25, 25)
    data = frame + angle_correction(camera_incl_angle)
    from matplotlib import pyplot as plt
    plt.figure()
    plt.imshow(data, origin='lower')
    plt.show()
    wind_velocity = 2
    """ Longwave radiation down is to be calculated from Stephan-Bolzman's law """
    F_long_down = None
    """ Shortwave radiation in [W / m**2], averaged value from Dship global_radiation measurements """
    F_short_down = 13
    """ Surface brightness temperatures, measured with the IR camera """
    T_surface = data
    snow_thickness = 0
    """ 2-meter air temperature, can be estimated with:
            * Dship data
            * temperature of a metal bar on the image
            * averaged surface temperature + offset (as it is done for satellite measurements by Yu and Rothrock [1996])
    """
    T_air = -5
    water_salinity = 31     # [ppt]
    """ Water temperature in [deg C], could be estimated from
            * Dship measurements
            * water freesing temperature for given water salinity
    """
    T_water = -1.8
    """ Relatime humidity from Dship measurements """
    relative_humidity = 0.9
    """ Fractional cloud cover """
    cloud_cover = 0

    ice_thickness, _, _ = calculate_ice_thickness(T_s=T_surface, T_w=T_water, T_a=T_air, h_s=snow_thickness, rh=relative_humidity,
                                            u=wind_velocity, S_w=water_salinity, F_ldn=F_long_down, F_sdn=F_short_down, C=cloud_cover)
#    [print('Ice thickness = {0} cm'.format(int(100 * item))) for item in ice_thickness[0]]
    plt.figure()
    plt.imshow(ice_thickness, origin='lower')
    plt.show()
    
    frame_norm = frame - frame.min()
    frame_norm /= frame_norm.max()
    segm = segment_image(frame_norm)
    bound = mark_boundaries(frame_norm, segm)
    plt.figure()
    plt.imshow(bound, origin='lower')
    plt.show()
    
    
    segment_medians = {}
    for segment in regionprops(segm, intensity_image=frame):
        segment_medians[segment.label] = np.ma.median(np.ma.masked_array(data=segment.intensity_image, mask=np.where(segment.intensity_image == 0, True, False)))
    

    merged = merge_neighbouring_segments(segm, segment_medians)
    bound_merged = mark_boundaries(frame_norm, merged)
    plt.figure()
    plt.imshow(bound_merged, origin='lower')
    plt.show()