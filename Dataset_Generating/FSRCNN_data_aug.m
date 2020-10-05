clear; 
%% To do data augmentation
folder = 'Train/HR/General-100';
savepath = 'Train/HR_aug/General-100_aug/';

filepaths = dir(fullfile(folder,'*.bmp'));
     
for i = 1 : length(filepaths)
    filename = filepaths(i).name;
    [add, im_name, type] = fileparts(filepaths(i).name);
    image = imread(fullfile(folder, filename));
    
    for angle = 0: 90 :270
        im_rot = imrotate(image, angle);
        %imwrite(im_rot, [savepath im_name, '_rot' num2str(angle) '.png']);
        
        for scale = 0.6 : 0.1 :1.0
            im_down = imresize(im_rot, scale, 'bicubic');
            imwrite(im_down, [savepath im_name, '_rot' num2str(angle) '_s' num2str(scale) '.png']);
        end
        
    end
end
