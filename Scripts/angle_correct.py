#! /usr/bin/env python

# Correction for emissivity fresnel equation
# and utility functions for computing the incidence angles and the surface areas seen by the pixels.s
from Fresnel_PenetrationDepth import dconstf, nfresnel
import numpy as np
import pylab as plt
C = 2.99708e8 # speed of light
F = C/7.6e-6 # Frequencies in Hz 7 \mu m
CAM_FOV = {10: np.array([57.1,44.4]), 20: np.array([30.4,23.1])} # deg 10mm or deg 20mm
CAM_IMG_SIZE = (640, 480)


def angles_tcam (cam_incl_ang, fov=10):
    img_cam_angles=np.zeros(CAM_IMG_SIZE) # Size of image

    size_img=np.array(CAM_IMG_SIZE)

    h_cam_angles_fov = CAM_FOV[fov]/2

    x_ang_range=np.linspace(-h_cam_angles_fov[0],h_cam_angles_fov[0],size_img[0])
    y_ang_range=np.linspace(90-(h_cam_angles_fov[1]+cam_incl_ang),90+h_cam_angles_fov[1]-cam_incl_ang,size_img[1])

    x_ang_range_rad=x_ang_range*np.pi/180
    y_ang_range_rad=y_ang_range*np.pi/180

    for i in xrange(size_img[0]):
        for j in xrange(size_img[1]):
            img_cam_angles[i,j]=np.arctan(np.tan(y_ang_range_rad[j])/np.cos(x_ang_range_rad[i]))

    return img_cam_angles.T


def em_angles_i(cam_incl_ang,t,s, f=F, fov=10):

    th = angles_tcam(cam_incl_ang, fov=fov)
    dconst = dconstf(f,t,s)
    [rh, rv] = nfresnel(th, dconst) # reflectance
    eh = 1. - rh 
    ev = 1. - rv 
    etot = (eh + ev)/2.

    return etot


def em_angles_t(cam_incl_ang,t,s, f=F, fov=10):
    e_tot=em_angles_i(cam_incl_ang,t,s, f=F, fov=fov)
    return e_tot**0.25


def areas_tcam(cam_incl_ang, height, fov=10):
    """Compute the area covered by each pixel.
    
    Suppose a setup where the camera has a angle from the horizontal. We approximate the projection of the pixels on the sea as parallelograms
    
    Args:
    cam_incl_ang - the inclination from the horizontal (degrees)
    height       - height where the camera is installed (metres)
    fov          - field of view of the cam (10 or 20, def. 10)
    """
    img_cam_areas = np.zeros(CAM_IMG_SIZE) # Size of image

    h_cam_angles_fov = CAM_FOV[fov]/2  # FOV in degrees
  
    x_ang_range = np.deg2rad(np.linspace(-h_cam_angles_fov[0], h_cam_angles_fov[0], img_cam_areas.shape[0] + 1))
    y_ang_range = np.deg2rad(np.linspace(h_cam_angles_fov[1] + cam_incl_ang, h_cam_angles_fov[1] - cam_incl_ang, img_cam_areas.shape[1] + 1))
  
    for i in range(1, x_ang_range.size - 1):
        for j in range(1, y_ang_range.size - 1):
            th_moy = (y_ang_range[j] + y_ang_range[j - 1]) / 2  # Angle of incidence in the middle of the pixel
            b = (height * np.tan(th_moy)) * (np.tan(x_ang_range[i]) - np.tan(x_ang_range[i - 1]))  # base of the parallelogram
            h = height * (np.tan(y_ang_range[j]) - np.tan(y_ang_range[j - 1]))  # height of the parallelogram
            img_cam_areas[i, j] = b * h
            
    return img_cam_angles.T
    

def project_image(cam_incl_ang, height, fov=10):
    """Compute the X and Y coordinates meshgrids to project the photos to a cartesian map.
    
    Args:
    cam_incl_ang -  Camera inclination from the horizontal (degrees)
    height       -  Height at which the focus spot of the camera is (meters)
    fov          -  Fov of the lens (10 or 20, mm) def: 10
    
    Returns two meshgrids, X and Y, in a 2D coordinate system where the camera is at 0,0.
    """
    h_cam_angles_fov = CAM_FOV[fov] / 2  # FOV in degrees
    phi_range = np.deg2rad(np.linspace(-h_cam_angles_fov[0], h_cam_angles_fov[0], CAM_IMG_SIZE[0]))  # Angles along the width of the image (X)
    theta_range = np.deg2rad(90 - cam_incl_ang + np.linspace(-h_cam_angles_fov[1], h_cam_angles_fov[1], CAM_IMG_SIZE[1]))  # Angles from the vertical (Y) 
    Phi, Theta = np.meshgrid(phi_range, theta_range, indexing='ij')
    
    # X is the distance from where the phi=0 ray touches the surface to where the phi ray lands.
    # Y is the distance from the vertical directly below the camera to where the ray lands.
    return ((height / np.cos(Theta)) * np.tan(Phi)).T , (height * np.tan(Theta)).T


if __name__ == '__main__':

    cam_incl_ang = 30  # deg
    cam_height = 25  # metres
    s = 32 # salinity in PPT
    t = -1.9# temperature in degC

    f = F

    img_cam_angles = angles_tcam (cam_incl_ang)

    e_tot=em_angles_i(cam_incl_ang,t,s,f)

    e_cor=em_angles_t(cam_incl_ang,t,s,f)

    img_areas = areas_tcam(cam_incl_ang, cam_height)
#    plt.figure(1)
#    plt.pcolormesh(img_cam_angles.T*180/np.pi)
#    plt.axis('scaled')
#    plt.colorbar()
#
#    plt.figure(2)
#    plt.pcolormesh(e_tot.T)
#    plt.axis('scaled')
#    plt.colorbar()
#
#    plt.figure(3)
#    plt.pcolormesh(e_cor.T)
#    plt.axis('scaled')
#    plt.colorbar()
#    plt.show()


    print(img_areas.sum())

    plt.figure(4)
    plt.imshow(img_areas, origin='lower')
    plt.colorbar()
    plt.show()



