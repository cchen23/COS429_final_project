%  Copyright (c) 2015, Omkar Parkhi
%  All rights reserved.
function server()


config.paths.net_path = '/Users/omkar/Work/bbc_face_match/backend/data/omkar_net.mat';
config.paths.face_model_path = '/Users/omkar/Work/bbc_face_match/backend/data/face_model.mat';
config.paths.feats_path = '/Users/omkar/Work/bbc_face_match/backend/data/YPfeats.mat';

config.server.port = 45303;
faceDet = lib.face_detector.dpmCascadeDetector(config.paths.face_model_path);
convNet = lib.face_feats.convNet(config.paths.net_path);
faceSim = lib.face_compare.simMeasure(config.paths.feats_path);



fprintf('Data Loading Complete\n');

fprintf('Waiting for connection\n');

t = tcpip('localhost',config.server.port,'NetworkRole','server');

fopen(t);
fprintf('Connected\n');
tcpStart = tic;
readData = false;
while 1

    if t.BytesAvailable>0
        readData = true;
        break;
    end
    
    if(toc(tcpStart)>5)
        break;
    end
end


outMsg = '';
dataComm = false;
imgRead = false;
detFace = false;
ftComp = false;
simDist = false;

if readData
    tic
    fprintf('Processing Request\n');
    try
        
        data = fread(t,t.BytesAvailable);
        data = char(data');
        tempele = regexp(data,':','split');
        fileName = tempele{2};
        fileName = strrep(fileName,'"','');
        fileName = strrep(fileName,'[','');
        fileName = strrep(fileName,'}','');
        fileName = strtrim(fileName);
        dataComm = true;
    catch
        dataComm = false;
        outMsg = '{"status":0,"output":[],"error":"Error in TCP communication"}';
    end
    if dataComm
        try
            img = imread(fileName);
            imgRead = true;
        catch
            imgRead = false;
            outMsg = '{"status":0,"output":[],"error":"Could not read image"}';
        end
        if imgRead
            try
                det = faceDet.detect(img);
                if(~isempty(det))
                    crop = lib.face_proc.faceCrop.crop(img,det(1:4,1));
                    detFace = true;
                else
                    detFace = false;
                    outMsg = '{"status":0;"output":[];"error":"Could not detect face"}';
                end
            catch
                detFace = false;
                outMsg = '{"status":0,"output":[],"error":"Could not detect face"}';
            end
            
            if detFace
                try
                    feat = convNet.simpleNN(crop);
                    ftComp = true;
                catch
                    ftComp = false;
                    outMsg = '{"status":0,"output":[],"error":"Could not compute features"}';
                end
                
                if ftComp
                    try
                        [neighbours,scores] = faceSim.findNNeighbours(feat,5);
                        outPutStr = '[';
                        for i=1:5
                            if i>1
                                outPutStr = sprintf('%s,',outPutStr);
                            end
                            outPutStr = sprintf('%s{"id":%d,"score":%0.2f}',outPutStr,...
                                neighbours(i),scores(i));
                            
                        end
                        outPutStr = sprintf('%s]',outPutStr);
                        outMsg = sprintf('{"status":1,"output":%s,"error":""}',outPutStr);
                    catch
                        outMsg = '{"status":0,"output":[],"error":"Couldnt find similar faces"}';
                    end
                end
            end
            
            
        end
    end
    toc
    
else
    outMsg = '{"status":0;"output":[];error:"Error in TCP communication"}';
end

fwrite(t,outMsg);
fclose(t);

end