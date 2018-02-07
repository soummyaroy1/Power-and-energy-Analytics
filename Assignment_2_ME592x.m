%% Preprocessing of variables
% Normalisation
load('MAINS.mat')
load('HVAC.mat')
load('LIGHTING.mat')
load('APPLIANCES.mat')
load('OTHER_LOADS.mat')
t = [1:262800]';
figure(1)

% Appliances
subplot(5,5,[1 3])      % Original Data
plot(y1)
title('Full Data Set Plots')
ylabel('Appl.')
ylim([-1 5])
xlim([0 2.5e5])

subplot(5,5,[4 5])     % Zoom In Data
plot(y1)
xlim([0 20000])
ylim([0 5])
title('Zoom In')

% HVAC
subplot(5,5,[6 8])      % Original Data
plot(y2)
ylabel('HVAC')
ylim([-2 5])
xlim([0 2.5e5])

subplot(5,5,[9 10])     % Zoom In Data
plot(y2)
xlim([0 20000])
ylim([-1 5])

% LIGHTING
subplot(5,5,[11 13])      % Original Data
plot(y3)
ylabel('LIGHTING')
ylim([-1 1])
xlim([0 2.5e5])

subplot(5,5,[14 15])     % Zoom In Data
plot(y3)
xlim([0 20000])
ylim([-1 1])

% MAINS
subplot(5,5,[16 18])      % Original Data
plot(y4)
ylabel('MAINS')
ylim([-1 1])
xlim([0 2.5e5])

subplot(5,5,[19 20])     % Zoom In Data
plot(y4)
xlim([0 20000])
ylim([-1 1])

% Other Loads
subplot(5,5,[21 23])      % Original Data
plot(y5)
ylabel('Others')
ylim([-1 1])
xlim([0 2.5e5])

subplot(5,5,[24 25])     % Zoom In Data
plot(y5)
xlim([0 20000])
ylim([-1 1])

%% Calculating minimum & maximum values in each data set
y1_min = min(y1);
y2_min = min(y2);
y3_min = min(y2);
y4_min = min(y4);
y5_min = min(y5);

y1_max = max(y1);
y2_max = max(y1);
y3_max = max(y3);
y4_max = max(y4);
y5_max = max(y5);

%% Calculating Standard Normal
y1_norm = (y1 - y1_min) ./ (y1_max - y1_min);
y2_norm = (y2 - y2_min) ./ (y2_max - y2_min);
y3_norm = (y3 - y3_min) ./ (y3_max - y3_min);
y4_norm = (y4 - y4_min) ./ (y4_max - y4_min);
y5_norm = (y5 - y5_min) ./ (y5_max - y5_min);

%% Plotting original data set along its Standard Normalization
figure(2)

% Appliances
subplot(5,4,[1 2])      % Original Data
plot(y1)
title('Full Data Set Plots')
ylabel('Appl.')
ylim([-1 5])
xlim([0 2.5e5])

subplot(5,4,[3 4])     % Standard Normal
plot(y1_norm)
xlim([0 20000])
xlim([0 2.5e5])
title('Standard Normalized')

% HVAC
subplot(5,4,[5 6])      % Original Data
plot(y2)
ylabel('HVAC')
ylim([-2 5])
xlim([0 2.5e5])

subplot(5,4,[7 8])     % Standard Normal
plot(y2_norm)
xlim([0 2.5e5])

% LIGHTING
subplot(5,4,[9 10])      % Original Data
plot(y3)
ylabel('LIGHTING')
ylim([-1 1])
xlim([0 2.5e5])

subplot(5,4,[11 12])     % Standard Normal
plot(y3_norm)
xlim([0 2.5e5])
%ylim([-1 1])

% MAINS
subplot(5,4,[13 14])      % Original Data
plot(y4)
ylabel('MAINS')
ylim([-1 1])
xlim([0 2.5e5])

subplot(5,4,[15 16])     % Standard Normal
plot(y4_norm)
xlim([0 2.5e5])
%ylim([-1 1])

% Other Loads
subplot(5,4,[17 18])      % Original Data
plot(y5)
ylabel('Others')
ylim([-1 1])
xlim([0 2.5e5])

subplot(5,4,[19 20])     % Standard Normal
plot(y5_norm)
xlim([0 2.5e5])

%% Denoising Parameters Calculation

len = length(y1);           % Commun data length

% Exploring the estimated 'lev' parameter given diff. Wavelet Families
wname_dmey = 'dmey';
wname_db = 'db10';
wname_bior = 'bior5.5';
wnamer_bio = 'rbio2.8';
wname_fk = 'fk8';

lev1 = wmaxlev(len, wname_dmey);
lev2 = wmaxlev(len, wname_dmey);
lev3 = wmaxlev(len, wname_dmey);
lev4 = wmaxlev(len, wname_dmey);
lev5 = wmaxlev(len, wname_dmey);

%lev4 = 10;

%% Denoising the Data Sets

% APPLIANCE
dnsig1 = wden(y1,'sqtwolog','s','mln',lev1,wname_dmey);

% HVAC
dnsig2 = wden(y2_norm,'sqtwolog','s','mln',lev2,wname_dmey);

% LIGHTING
dnsig3 = wden(y3_norm,'sqtwolog','s','mln',lev3,wname_dmey);

% MAINS
dnsig4 = wden(y4_norm,'sqtwolog','s','mln',lev4,wname_dmey);

% OTHER LOADS
dnsig5 = wden(y5_norm,'sqtwolog','s','mln',lev5,wname_dmey);

%% Plotting Original and Denoised Time Series
figure(3)

% Appliances
subplot(5,1,1)      
hold on
plot(y1)
plot(dnsig1, 'r')
ylabel('Appl.')
ylim([-1 5])
xlim([0.5e5 1.5e5])
hold off

% HVAC
subplot(5,1,2) 
hold on
plot(y2)
plot(dnsig2, 'r')
ylabel('HVAC')
ylim([-1 3])
xlim([0.5e5 1.5e5])
hold off

% LIGHTING
subplot(5,1,3) 
hold on
plot(y3)
plot(dnsig3, 'r')
ylabel('LIGHTING')
ylim([-0.2 0.8])
xlim([0.5e5 1.5e5])
hold off

% MAINS
subplot(5,1,4) 
hold on
plot(y4)
plot(dnsig4, 'r')
ylabel('MAINS')
ylim([-0.25 0.8])
xlim([0.5e5 1.5e5])
hold off

% Other Loads
subplot(5,1,5) 
hold on
plot(y5)
plot(dnsig5, 'r')
ylabel('OTHERS')
ylim([0 0.8])
xlim([0.5e5 1.5e5])
hold off
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
plot(t,y1)
hold on;
plot(t,y2)

%% Convert data to frequency domain and compare Euclidean Distance
d = [y1 y2 y3 y4 y5];
Y = fft(d);
ED12 = sum((Y(:,1) - Y(:,2)).^2).^0.5;
ED13 = sum((Y(:,1) - Y(:,3)).^2).^0.5;
ED14 = sum((Y(:,1) - Y(:,4)).^2).^0.5;
ED15 = sum((Y(:,1) - Y(:,5)).^2).^0.5;
ED = [ED12;ED13;ED14;ED15];
min(ED) % ED between MAINS and HVAC is minimum even in frequency domain
max(ED) % ED between MAINS and APPLIANCES is maximum in frequency domain ( WHY?)
%plot(t,Y(:,1))
%hold on;
%plot(t,Y(:,2))
plot(abs(Y)) %f you do nothing to the original signal, then the amplitude of the FFT is of the same units as your original signal.

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

ge1 = etfe(y1,[],32);
ge2 = etfe(y1,[],512);
spectrum(ge1,'b.-',ge2,'g')

[pxx,w] = periodogram(y1);
plot(w,10*log10(pxx))

%% Sampling rate in fft
Fs = 1;
t = 0:1/Fs:1;
x = Y(:,1);
xdft = fft(x);
xdft = xdft(1:length(x)/2+1);
DF = Fs/length(x);
freqvec = 0:DF:Fs/2;
plot(freqvec,abs(xdft))


