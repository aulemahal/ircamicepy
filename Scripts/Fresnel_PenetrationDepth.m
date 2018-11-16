% Fresnel equations and penetration depth
% 
% from Lars Kaleschke's seawater.pro (IDL)
% 2018-09 Gunnar Spreen

% Equations for Calculating the Dielectric Constant of Saline Water
% A. Stogryn, IEEE Trans. on MW Theory and Techniques, August 1971

plotfile = '/Users/gspreen/Seafile/Seafile/Campaigns/2018-09 Polarstern PS 115-2/Infrared Camera/Literature/FresnelWater_MW+IR.png';

s = 35; % salinity in PPT
t = 0; % temperature in °C
thdeg = 0:90; % incidence angle
th = thdeg/360*2*pi; 
c = 2.99708E8; % speed of light

f37 = 10E9; % MW frequency in Hz
fir = c/10E-6; % IR frequency in Hz
f500 = c/500E-9; % visible light (green) frequency in Hz

% 37 GHz
dconst37 = dconstf(f37,t,s);
[rh37, rv37] = nfresnel(th, dconst37); % reflectance
eh37 = 1 - rh37; ev37 = 1 - rv37; % emssivity

% IR 10 Micrometer
dconstir = dconstf(fir,t,s);
[rhir, rvir] = nfresnel(th, dconstir); % reflectance
ehir = 1 - rhir; evir = 1 - rvir; % emssivity
etotir = (ehir + evir)/2;

% Vis 500 Nanometer
dconst500 = dconstf(f500,t,s);
[rh500, rv500] = nfresnel(th, dconst500); % reflectance
eh500 = 1 - rh500; ev500 = 1 - rv500; % emssivity
etot500 = (eh500 + ev500)/2;

figure(1); clf 
plot(thdeg,eh37);
hold on
plot(thdeg,ev37)
plot(thdeg,etotir,'LineWidth',2);
plot(thdeg,ehir);
plot(thdeg,evir);
% plot(thdeg,etot500,'LineWidth',2);
% plot(thdeg,eh500);
% plot(thdeg,ev500);
hold off
xlabel('Incidence Angle [°]','FontSize',12)
ylabel('Emissivity','FontSize',12)
legend('37 GHz H', '37 GHz V', '10 μm', '10 μm H', '10 μm V', ...
    'Location','SouthWest')
%    '500 nm', '500 nm H', '500 nm V', ...
 print('-f1', '-dpng', '-r200', plotfile)

function a = af(n)
a = 1.0 - 0.2551 * n + 5.151E-2 * n^2 - 6.889E-3 * n^3;
end

function b = bf(n,t)
b = 0.1463E-2 * n * t + 1 - 0.04896 * n - 0.02967 * n^2 + 5.644E-3 * n^3;
end

function e0 = e0f(t)
e0 = 87.74 - 4.0008 * t + 9.398E-4 * t^2 + 1.41E-6*t^3;
end

function etn = etnf(t,n)
etn = e0f(t) * af(n);
end

function estat = estatf(t,s)
estat = 81.82 + (-6.05E-2 + (-3.166E-2 + (3.109E-3  + (-1.179E-4 + 1.483E-6 * t)*t)*t)*t)*t - s*(0.1254 + (9.403E-3 + (-9.555E-4 +(9.088E-5 + (-3.601E-6 + 4.713E-8*t)*t)*t)*t)*t);
end

function relax = calc_relax(t)
% relax = 2 pi \tau
relax = 1.1109E-10 - 3.824E-12 * t + 6.938E-14 * t^2 - 5.096E-16 * t^3;
end

function rel = relaxf(t,n)
rel = calc_relax(t) * bf(n,t);
end

function n = nf(s)
% s: salinity in ppt, valid for 0-260
n =  s * (1.707E-2 + 1.205E-5 * s + 4.058E-9 * s^2);
end

function sig = sigseawater25(s)
% Ionic conductivity of sea water at a temperature of 25 C
% 0 < s < 40 ppt
sig = s * (0.182521 - 1.46192E-3 * s + 2.09324E-5 * s^2 - 1.28205E-7 * s^3);
end

function ioncond = ioncondf(t,s)
% Ionic conductivity
% Frequency dependent (Debye-Falkenhagen effect!)
d = 25.0-t;
al = 2.033E-2 + 1.266E-4 * d + 2.464E-6 * d^2 - s * (1.849E-5 - 2.551E-7 * d + 2.551E-8 * d^2);
ioncond = sigseawater25(s) * exp(-d * al);
end

%Dielectrical constant of water
function dconst = dconstf(f,t,s)
% t in C
% s in ppt
e00 = 4.9;
es0 = 8.854E-12;
n = nf(s);
%dconst = e00+(etn(t,n)-e00)/(1-complex(0,relaxf(t,n)*f))+complex(0,ioncond(t,s)/(2*!pi*es0*f))
dconst = e00 + (estatf(t,s) - e00)/(1-complex(0,relaxf(t,n)*f)) + complex(0,ioncondf(t,s)/(2*pi*es0*f));
end

% function nfresnel,th,k
% p=1/sqrt(2)*(((float(k)-sin(th)^2)+imaginary(k)^2.0)^.5+(float(k)-sin(th)^2.0))^.5
% q=1/sqrt(2)*(((float(k)-sin(th)^2)+imaginary(k)^2.0)^.5-(float(k)-sin(th)^2.0))^.5
% rh=((p-cos(th))^2.0+q^2.0)/((p+cos(th))^2.0+q^2.0)
% rv=((float(k)*cos(th)-p)^2.0+(imaginary(k)*cos(th)-q)^2.0)/((float(k)*cos(th)+p)^2.0+(imaginary(k)*cos(th)+q)^2.0)
% end
function [rh, rv] = nfresnel(th,dconst)
rh = (cos(th) - sqrt(dconst-sin(th).^2))./(cos(th) + sqrt(dconst-sin(th).^2));
rv = (dconst*cos(th) - sqrt(dconst-sin(th).^2))./(dconst*cos(th) + sqrt(dconst-sin(th).^2));
rh = abs(rh).^2;
rv = abs(rv).^2;
end
