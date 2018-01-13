%  Copyright (c) 2015, Omkar M. Parkhi
%  All rights reserved.

config.paths.net_path = 'data/vgg_face.mat';
config.paths.face_model_path = 'data/face_model.mat';

faceDet = lib.face_detector.dpmCascadeDetector(config.paths.face_model_path);
convNet = lib.face_feats.convNet(config.paths.net_path);



img = imread('ak.jpg');
det = faceDet.detect(img);
crop = lib.face_proc.faceCrop.crop(img,det(1:4,1));
[score,class] = max(convNet.simpleNN(crop));
