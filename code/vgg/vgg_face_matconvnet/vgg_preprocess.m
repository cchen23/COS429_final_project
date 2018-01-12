%  Preprocess LFW data for VGG

config.paths.net_path = 'data/vgg_face.mat';
config.paths.face_model_path = 'data/face_model.mat';

faceDet = lib.face_detector.dpmCascadeDetector(config.paths.face_model_path);
convNet = lib.face_feats.convNet(config.paths.net_path);

faces = load('faces.mat');
faces_data = faces.data;
s = size(faces_data); % Should be n x 250 x 250 x 3

img = uint8(squeeze(faces_data(1,:,:,:)));
imshow(img);
imwrite(img, 'original_image.png');

mat = zeros(s(1), 224, 224 ,3);
for i=1:s(1)
    i
    
    img = uint8(squeeze(faces_data(i,:,:,:)));
    det = faceDet.detect(img);
    
    det_size = size(det);
    if (det_size(1) > 0 && det_size(2) > 0)
        crop = lib.face_proc.faceCrop.crop(img,det(1:4,1));
    else
        crop = img;
    end
    
    if sum(size(crop) == [224 224 3]) < 3
        crop = imresize(crop, [224, 224]);
    end
    
    mat(i,:,:,:) = crop;
    % size(crop)
    % imshow(crop);
    % imwrite(img, 'processed_image.png');
end

size(mat)
save('preprocessed_faces.mat', 'mat', '-v7.3');

img = uint8(squeeze(mat(1,:,:,:)));
imshow(img);
