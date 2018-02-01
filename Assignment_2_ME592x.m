%% Preprocessing of variables
% Normalisation
load('MAINS.mat')
load('HVAC.mat')
load('LIGHTING.mat')
load('APPLIANCES.mat')
load('OTHER_LOADS.mat')
t = [1:262800]';
y1 = (y1 - mean(y1))/(max(y1)-min(y1));
y2 = (y2 - mean(y2))/(max(y2)-min(y2));
y3 = (y3 - mean(y3))/(max(y3)-min(y3));
y4 = (y4 - mean(y4))/(max(y4)-min(y4));
y5 = (y5 - mean(y5))/(max(y5)-min(y5));

% Denoising
% 'sqtwolog' for the universal threshold G
% sqrt(2ln(·)), ‘s’ is for soft thresholding, 
% ‘mln’ is for rescaling done using level dependant estimation, 13s is the level of denoising, 
% ‘symN’ is name of wavelet denoising technique,'coifN' stands for
% coiflet wavelet denoising.
[y1d,c1,l1] = wden(y1,'sqtwolog','s','mln',13,'sym14'); % wmaxlev(size(y1),'sym14') to find maximum level of wavelet decompostion
[y2d,c2,l2] = wden(y2,'sqtwolog','s','mln',11,'dmey'); 
[y3d,c3,l3] = wden(y3,'sqtwolog','s','mln',16,'sym2');
[y4d,c4,l4] = wden(y4,'sqtwolog','s','mln',18,'db1');
[y5d,c5,l5] = wden(y5,'sqtwolog','s','mln',14,'coif2');

% Plotting the normalised and denoised time series and comparing
figure(1)
subplot(2,5,1)
plot(t,y1) % Normalised Data for MAINS
title('MAINS')
subplot(2,5,6)
plot(t,y1d) % Denoised Data for MAINS
title('Symlet 14 ')
subplot(2,5,2)
plot(t,y2)
title('HVAC')
subplot(2,5,7)
plot(t,y2d)
title('Discrete Meyer')
subplot(2,5,3)
plot(t,y3)
title('LIGHTING')
subplot(2,5,8)
plot(t,y3d)
title('Symlet 2')
subplot(2,5,4)
plot(t,y4)
title('APPLIANCES')
subplot(2,5,9)
plot(t,y4d)
title('Daubechies1') % db1 or Haar Wavelet transform is useful for symmetric data edge point detection
subplot(2,5,5)
plot(t,y5)
title('OTHER LOADS')
subplot(2,5,10)
plot(t,y5d)
title('coiflet 2')

%% Calculating Direct Euclidean Distance

ed12 = sum((y1 - y2).^2).^0.5;
ed13 = sum((y1 - y3).^2).^0.5;
ed14 = sum((y1 - y4).^2).^0.5;
ed15 = sum((y1 - y5).^2).^0.5;
ed = [ed12;ed13;ed14;ed15];
min(ed) % ED between signals 1 and 2 i.e MAINS and HVAC is minimum
max(ed) % ED between signals 1 and 3 i.e MAINS and LIGHTING is maximum

%% Convert data to frequency domain and compare Euclidean Distance
% Ask TA what exactly Dr.Sarkar wants in the comparison questions..

d = [y1 y2 y3 y4 y5];
Y = fft(d);
ED12 = sum((Y(:,1) - Y(:,2)).^2).^0.5;
ED13 = sum((Y(:,1) - Y(:,3)).^2).^0.5;
ED14 = sum((Y(:,1) - Y(:,4)).^2).^0.5;
ED15 = sum((Y(:,1) - Y(:,5)).^2).^0.5;
ED = [ED12;ED13;ED14;ED15];
min(ED) % ED between MAINS and HVAC is minimum even in frequency domain
max(ED) % ED between MAINS and APPLIANCES is maximum in frequency domain ( WHY?)

%% Use KL Divergence metric and perform comparison

pvect1 = Y(:,1);
pvect2 = Y(:,2);
pvect3 = Y(:,3);
pvect4 = Y(:,4);
pvect5 = Y(:,5);
KL1 = sum(pvect1.*(log2(pvect1)-log2(pvect2)));
KL2 = sum(pvect1.*(log2(pvect1)-log2(pvect3)));
KL3 = sum(pvect1.*(log2(pvect1)-log2(pvect4)));
KL4 = sum(pvect1.*(log2(pvect1)-log2(pvect5)));
KL = [KL1;KL2;KL3;KL4];
min(KL) % KL Divergence metric is minimum between MAINS and HVAC
max(KL) % KL Divergence metric is maximum between MAINS and APPLIANCES

%% Compare data in Wavelet Transformed Space[TBS]

%% Use windowed spectrogram to identify motifs in the main power data to detect changes in time-series characteristics

s = spectrogram(y1);








