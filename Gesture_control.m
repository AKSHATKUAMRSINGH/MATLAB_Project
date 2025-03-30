function gesturerecognisition(redThresh, greenThresh, blueThresh, numFrame)

    warning('off','vision:transition:usesOldCoordinates');
    
    
    if nargin < 1
    
        redThresh = 0.22; 
    
        greenThresh = 0.14;
    
        blueThresh = 0.18;
    
        numFrame = 1000;
    
    end
    
    
    cam = imaqhwinfo;
    
    cameraName = char(cam.InstalledAdaptors(end));
    
    cameraInfo = imaqhwinfo(cameraName);
    
    cameraId = cameraInfo.DeviceInfo.DeviceID(end);
    
    cameraFormat = char(cameraInfo.DeviceInfo.SupportedFormats(end));
    
    
    jRobot = java.awt.Robot;
    
    vidDevice = imaq.VideoDevice(cameraName, cameraId, cameraFormat, ...
    
                        'ReturnedColorSpace', 'RGB');
    
    
    vidInfo = imaqhwinfo(vidDevice); 
    
    screenSize = get(0,'ScreenSize');
    
    hblob = vision.BlobAnalysis('AreaOutputPort', false, ...
    
                                    'CentroidOutputPort', true, ...
    
                                    'BoundingBoxOutputPort', true', ...
    
                                    'MaximumBlobArea', 3000, ...
    
                                    'MinimumBlobArea', 100, ...
    
                                    'MaximumCount', 3);
    
    hshapeinsBox = vision.ShapeInserter('BorderColorSource', 'Input port', ...
    
                                        'Fill', true, ...
    
                                        'FillColorSource', 'Input port', ...
    
                                        'Opacity', 0.4);
    
    hVideoIn = vision.VideoPlayer('Name', 'Final Video', ...
    
                                    'Position', [100 100 vidInfo.MaxWidth+20 vidInfo.MaxHeight+30]);
    
    nFrame = 0;
    
    lCount = 0; rCount = 0; dCount = 0;
    
    sureEvent = 5;
    
    iPos = vidInfo.MaxWidth/2;
    
    
    
    while (nFrame < numFrame)
    
        rgbFrame = step(vidDevice);
    
        rgbFrame = flipdim(rgbFrame,2);
    
        diffFrameRed = imsubtract(rgbFrame(:,:,1), rgb2gray(rgbFrame));
    
        binFrameRed = im2bw(diffFrameRed, redThresh);
    
        [centroidRed, bboxRed] = step(hblob, binFrameRed);
    
    
        diffFrameGreen = imsubtract(rgbFrame(:,:,2), rgb2gray(rgbFrame));
    
        binFrameGreen = im2bw(diffFrameGreen, greenThresh);
    
        [centroidGreen, bboxGreen] = step(hblob, binFrameGreen);
    
       
    
        diffFrameBlue = imsubtract(rgbFrame(:,:,3), rgb2gray(rgbFrame));
    
        binFrameBlue = im2bw(diffFrameBlue, blueThresh);
    
        [~, bboxBlue] = step(hblob, binFrameBlue);
    
       
    
        if length(bboxRed(:,1)) == 1
    
            jRobot.mouseMove(1.5*centroidRed(:,1)*screenSize(3)/vidInfo.MaxWidth, 1.5*centroidRed(:,2)*screenSize(4)/vidInfo.MaxHeight);
    
        end
    
        if ~isempty(bboxBlue(:,1))
    
            if length(bboxBlue(:,1)) == 1
    
                lCount = lCount + 1;
    
                if lCount == sureEvent
    
                    jRobot.mousePress(16);
    
                    pause(0.1);
    
                    jRobot.mouseRelease(16);
    
                end
    
            elseif length(bboxBlue(:,1)) == 2
    
                rCount = rCount + 1;
    
                if rCount == sureEvent
    
                    jRobot.mousePress(4);
    
                    pause(0.1);
    
                    jRobot.mouseRelease(4);
    
                end
    
            elseif length(bboxBlue(:,1)) == 3
    
                dCount = dCount + 1;
    
                if dCount == sureEvent
    
                    jRobot.mousePress(16);
    
                    pause(0.1);
    
                    jRobot.mouseRelease(16);
    
                    pause(0.2);
    
                    jRobot.mousePress(16);
    
                    pause(0.1);
    
                    jRobot.mouseRelease(16);
    
                end
    
            end
    
        else
    
            lCount = 0; rCount = 0; dCount = 0;
    
        end
    
        if ~isempty(bboxGreen(:,1))
    
            if (mean(centroidGreen(:,2)) - iPos) < -2
    
                jRobot.mouseWheel(-1);
    
            elseif (mean(centroidGreen(:,2)) - iPos) > 2
    
                jRobot.mouseWheel(1);
    
            end
    
            iPos = mean(centroidGreen(:,2));
    
        end
    
        vidIn = step(hshapeinsBox, rgbFrame, bboxRed,single([1 0 0]));
    
        vidIn = step(hshapeinsBox, vidIn, bboxGreen,single([0 1 0]));
    
        vidIn = step(hshapeinsBox, vidIn, bboxBlue,single([0 0 1]));
    
        step(hVideoIn, vidIn);
    
        nFrame = nFrame+1;
    
    end
    
    
    release(hVideoIn);
    
    release(vidDevice);
    
    clc;
    
    end