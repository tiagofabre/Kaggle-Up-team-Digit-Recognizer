function images = loadMNISTImages(filename)
%loadMNISTImages returns a 28x28x[number of MNIST images] matrix containing
%the raw MNIST images

fp = fopen(filename, 'rb');
assert(fp ~= -1, ['Could not open ', filename, '']);

magic = fread(fp, 1, 'int32', 0, 'ieee-be');
assert(magic == 2051, ['Bad magic number in ', filename, '']);

numImages = fread(fp, 1, 'int32', 0, 'ieee-be');
numRows = fread(fp, 1, 'int32', 0, 'ieee-be');
numCols = fread(fp, 1, 'int32', 0, 'ieee-be');

images = fread(fp, inf, 'unsigned char');
images = reshape(images, numCols, numRows, numImages);
images = permute(images,[2 1 3]);

fclose(fp);

columns = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0];
imagesSimple = zeros(28,10000);
%converting to signal
for i=1:numImages
    
    H = fspecial('gaussian',[4 4],4);
    blurred = imfilter(images(:,:,i),H,'same');
    number = blurred;
    
    for section=0:2:26
        for l=1:28
            for c=1:2
                cc = c+section;
                if(number(l,cc) > 0)
                    columns((section+2)/2) = columns((section+2)/2) + (number(l,cc)/255);
                end    
            end
        end
        columns((section+2)/2) = columns((section+2)/2)/60;
    end
    
    for section=0:2:26
        for l=1:28
            for c=1:2
                cc = c+section;
                if(number(cc,l) > 0)
                    columns(((section+2)/2)+14) = columns(((section+2)/2)+14) + (number(cc,l)/255);
                end    
            end
        end
        columns(((section+2)/2)+14) = columns(((section+2)/2)+14)/60;
    end
    imagesSimple(:,i) = columns;
end


% Reshape to #pixels x #examples
%images = reshape(images, size(images, 1) * size(images, 2), size(images, 3));
% Convert to double and rescale to [0,1]
%images = double(images) / 255;
images = imagesSimple;

end
