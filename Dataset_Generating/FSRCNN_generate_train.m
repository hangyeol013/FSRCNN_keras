clear;close all;
%% settings
%% ÇÒ¶§¸¶´Ù folder À§Ä¡ test¶û train ¹Ù²ãÁà¾ß ÇÑ´Ù.
folder = 'Train/General-100_aug';
%folder = 'Train/BSD200';
savepath_test = 'cross_val_200_4.h5';
savepath_train = 'train_General100aug_2.h5';

size_input = 10;
size_label = 20;
scale = 2;
stride = 4;

%% initialization
data = zeros(size_input, size_input, 1, 1);
label = zeros(size_label, size_label, 1, 1);
padding = abs(size_input - size_label)/2;
count = 0;

%% generate data
filepaths = dir(fullfile(folder,'*.bmp'));
    
for i = 1 : length(filepaths)
    
    image = imread(fullfile(folder,filepaths(i).name));
    image = rgb2ycbcr(image);
    image = im2double(image(:, :, 1));
    
    im_label = modcrop(image, scale);
    im_input = imresize(im_label,1/scale,'bicubic');
    [hei,wid] = size(im_input);

    for x = 1 : stride : hei-size_input+1
        for y = 1 : stride : wid-size_input+1
            
            subim_input = im_input(x : x+size_input-1, y : y+size_input-1);
            subim_label = im_label((x-1)*scale + 1 : (x-1)*scale + size_label, (y-1)*scale + 1 : (y-1)*scale + size_label);

            count=count+1;
            data(:, :, 1, count) = subim_input;
            label(:, :, 1, count) = subim_label;
        end
    end
end

data_dims = size(data);
label_dims = size(label);

%% writing to HDF5

h5create(savepath_train,'/data',data_dims);
h5create(savepath_train, '/label', label_dims);

h5write(savepath_train, '/data', data);
h5write(savepath_train, '/label', label);

h5disp(savepath_train);
