%% Example script to display the ant world and ant route
% Paul Ardin, February 2014

clear all;clc;close all

% load ant world
load('world5000_gray.mat');

% load ant route data, 
load('AntRoutes.mat');

% draw the ant world in map view
 fill(X',Y','k');
 axis square;
 axis off;
 
% draw the ant route, using Ant 1 Route 1
hold on;
scatter(Ant1_Route1(:,1)/100,Ant1_Route1(:,2)/100,'.'); % Note the conversion of Route data to metres.

img = ImgGrabber_2013(Ant1_Route1(1,1)/100,Ant1_Route1(1,2)/100,0.01,Ant1_Route1(1,3)/100,X,Y,Z,colp,296,4);

%figure;imshow(img)

% Load test image.
load('test_img.mat');

 size(img)
 size(test_img)

% % Compare to make sure images are the same (should output 0)
 sumsqr(test_img-img)