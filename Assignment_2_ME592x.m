%% Preprocessing of variables
% Normalisation
load('MAINS.mat')
load('HVAC.mat')
load('LIGHTING.mat')
load('APPLIANCES.mat')
load('OTHER_LOADS.mat')
t = [1:262800]';
y1 = (y1 - mean(y1))/(max(y1)-min(y1));
plot(t,y1)
