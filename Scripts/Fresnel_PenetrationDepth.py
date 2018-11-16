# Fresnel equations and penetration depth
# 
# from Lars Kaleschke's seawater.pro (IDL)
# From 2018-09 Gunnar Spreen's Fresnel_PenetrationDepth.m
# 2018-09 Damien Ringeisen

# Equations for Calculating the Dielectric Constant of Saline Water
# A. Stogryn, IEEE Trans. on MW Theory and Techniques, August 1971

import numpy as np
import pylab as plt

def af(n):
    a = 1.0 - 0.2551 * n + 5.151e-2 * n**2 - 6.889e-3 * n**3
    return a

def bf(n,t) :
    b = 0.1463e-2 * n * t + 1 - 0.04896 * n - 0.02967 * n**2 + 5.644e-3 * n**3
    return b

def e0f(t) :
    e0 = 87.74 - 4.0008 * t + 9.398e-4 * t**2 + 1.41e-6*t**3
    return e0

def etnf(t,n) :
    etn = e0f(t) * af(n)
    return etn

def estatf(t,s) :
    estat = 81.82 + (-6.05e-2 + (-3.166e-2 + (3.109e-3  + (-1.179e-4 + 1.483e-6 * t)*t)*t)*t)*t - s*(0.1254 + (9.403e-3 + (-9.555e-4 +(9.088e-5 + (-3.601e-6 + 4.713e-8*t)*t)*t)*t)*t)
    return estat

def calc_relax(t):
    # relax = 2 pi \tau
    relax = 1.1109e-10 - 3.824e-12 * t + 6.938e-14 * t**2 - 5.096e-16 * t**3
    return relax

def relaxf(t,n) :
    rel = calc_relax(t) * bf(n,t)
    return rel

def nf(s) :
    # s: salinity in ppt, valid for 0-260
    n =  s * (1.707e-2 + 1.205e-5 * s + 4.058e-9 * s**2)
    return n

def sigseawater25(s) :
    # Ionic conductivity of sea water at a temperature of 25 C
    # 0 < s < 40 ppt
    sig = s * (0.182521 - 1.46192e-3 * s + 2.09324e-5 * s**2 - 1.28205e-7 * s**3)
    return sig

def  ioncondf(t,s) :
    # Ionic conductivity
    # Frequency dependent (Debye-Falkenhagen effect!)
    d = 25.0-t
    al = 2.033e-2 + 1.266e-4 * d + 2.464e-6 * d**2 - s * (1.849e-5 - 2.551e-7 * d + 2.551e-8 * d**2)
    ioncond = sigseawater25(s) * np.exp(-d * al)
    return ioncond

#Dielectrical constant of water
def dconstf(f,t,s) :
    # t in C
    # s in ppt
    e00 = 4.9
    es0 = 8.854e-12
    n = nf(s)
    #dconst = e00+(etn(t,n)-e00)/(1-complex(0,relaxf(t,n)*f))+complex(0,ioncond(t,s)/(2*!pi*es0*f))
    dconst = e00 + (estatf(t,s) - e00)/(1-complex(0,relaxf(t,n)*f)) + complex(0,ioncondf(t,s)/(2*np.pi*es0*f))
    return dconst

# def nfresnel,th,k
# p=1/sqrt(2)*(((float(k)-sin(th)**2)+imaginary(k)**2.0)**.5+(float(k)-sin(th)**2.0))**.5
# q=1/sqrt(2)*(((float(k)-sin(th)**2)+imaginary(k)**2.0)**.5-(float(k)-sin(th)**2.0))**.5
# rh=((p-cos(th))**2.0+q**2.0)/((p+cos(th))**2.0+q**2.0)
# rv=((float(k)*cos(th)-p)**2.0+(imaginary(k)*cos(th)-q)**2.0)/((float(k)*cos(th)+p)**2.0+(imaginary(k)*cos(th)+q)**2.0)
#   return

def nfresnel(th,dconst) :
    rh = (np.cos(th) - np.sqrt(dconst-np.sin(th)**2))/(np.cos(th) + np.sqrt(dconst-np.sin(th)**2))
    rv = (dconst*np.cos(th) - np.sqrt(dconst-np.sin(th)**2))/(dconst*np.cos(th) + np.sqrt(dconst-np.sin(th)**2))
    rh = np.abs(rh)**2
    rv = np.abs(rv)**2
    return [rh, rv]

if __name__=='__main__':

    plotfile = './FresnelWater_MW+IR.png'

    s = 34 # salinity in PPT
    t = -1.9# temperature in degC

    thdeg = np.arange(0,90,0.1) # incidence angle
    th = thdeg/360.*2.*np.pi 
    c = 2.99708e8 # speed of light

    fs={'100 $\mu m$':c/100e-6} # Dictionary of frequencies in Hz

    plt.figure(1)

    dconst={}
    rh={}
    rv={}
    ev={}
    eh={}
    etot={}

    for i in fs :

        istr=str(i)
        dconst[i] = dconstf(fs[i],t,s)
        [rh[i], rv[i]] = nfresnel(th, dconst[i]) # reflectance
        eh[i] = 1. - rh[i] 
        ev[i] = 1. - rv[i] # emissivity
        etot[i] = (eh[i] + ev[i])/2.

        plt.plot(thdeg,eh[i],label=i+' H')
        plt.plot(thdeg,ev[i],label=i+' V')
        plt.plot(thdeg,etot[i],linewidth=3,label=i+' T')


    plt.xlabel('Incidence Angle [$^\circ$]')
    plt.ylabel('emissivity')
    plt.legend()
    plt.grid()

    plt.savefig(plotfile,dpi=140)

