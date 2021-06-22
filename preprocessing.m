
%% Processing all subjects

originaldata_path = '/SAN/medic/Verdict/innovate_data/dataset/INNOVATE/';
dirListing = dir(originaldata_path);
addpath('MATLAB/nifti')
h = waitbar(0,'Processing all subjects');

patientname = ['INN-104-RWB/']
dir([originaldata_path patientname 'INN*'])

%% Run FSL and MRTrix for preprocessing 

tmp = dir([originaldata_path patientname 'INN*']);
subfoldername = [tmp.name '/'];

fprintf('cek b90 ....');
%tmp = dir([originaldata_path patientname subfoldername '*b90_*']);
tmp = dir([originaldata_path patientname subfoldername '*b90_*']);
files = dir(fullfile([originaldata_path patientname subfoldername '/' tmp(1).name '/NIFTI/'],'*.nii.gz'));
b90folder = files.name;
tmp = load_untouch_nii([files.folder '/' files.name]);
b90img = tmp.img;


tmp = dir([originaldata_path patientname subfoldername '*b500_*']);
files = dir(fullfile([originaldata_path patientname subfoldername '/' tmp(1).name '/NIFTI/'],'*.nii.gz'));
b500folder = files.name;
tmp = load_untouch_nii([files.folder '/' files.name]);
b500img = tmp.img;


tmp = dir([originaldata_path patientname subfoldername '*b1500_*']);
files = dir(fullfile([originaldata_path patientname subfoldername '/' tmp(1).name '/NIFTI/'],'*.nii.gz'));
b1500folder = files.name;
tmp = load_untouch_nii([files.folder '/' files.name]);
b1500img = tmp.img;

fprintf('cek b2000 ....');
tmp = dir([originaldata_path patientname subfoldername '*b2000_*']);
files = dir(fullfile([originaldata_path patientname subfoldername '/' tmp(1).name '/NIFTI/'],'*.nii.gz'));
b2000folder = files.name;
tmp = load_untouch_nii([files.folder '/' files.name]);
b2000img = tmp.img;

fprintf('cek b3000 ....');
tmp = dir([originaldata_path patientname subfoldername '*b3000_*']);
files = dir(fullfile([originaldata_path patientname subfoldername '/' tmp(1).name '/NIFTI/'],'*.nii.gz'));
b3000folder = files.name;
tmp = load_untouch_nii([files.folder '/' files.name]);
b3000img = tmp.img;

merged_img = cat(4, b90img, b500img, b1500img, b2000img, b3000img);

tmp.img = merged_img;
tmp.hdr.dime.dim(5) = size(merged_img,4);

%% Prepare pre-processing bash script

foldername = [patientname 'processed/'];

mkdir(foldername);

save_untouch_nii(tmp,[foldername 'rawdata_merged.nii.gz']);

fid = fopen([foldername 'pre_processing.sh'],'w');

fprintf(fid, '#!/bin/sh\n');
fprintf(fid,'. /share/apps/fsl-6.0.1/etc/fslconf/fsl.sh\n');
fprintf(fid,['/share/apps/fsl-6.0.1/bin/fslmerge -t ' foldername 'rawdata_merged.nii.gz ' originaldata_path patientname subfoldername b90folder '/NIFTI/*.nii.gz ' originaldata_path patientname subfoldername b500folder '/NIFTI/*.nii.gz ' originaldata_path patientname subfoldername b1500folder '/NIFTI/*.nii.gz ' originaldata_path patientname subfoldername b2000folder '/NIFTI/*.nii.gz ' originaldata_path patientname subfoldername b3000folder '/NIFTI/*.nii.gz\n']);
fprintf(fid,[' /share/apps/mrtrix3/bin/dwidenoise -force -extent 5,5,5 ' foldername 'rawdata_merged.nii.gz ' foldername 'rawdata_merged_deno.nii.gz']);


fclose(fid);

command = ['sh ' foldername 'pre_processing.sh'];

fprintf('Running MP-PCA denoising...\n')

[~,~] = system(command);

%% Loading denoised data

tmp = load_untouch_nii([foldername 'rawdata_merged_deno.nii.gz']);
img = tmp.img;

% %% Gibbs derung

fprintf('Running Gibbs ringing correction...\n')

mask = ones(size(img));

[img_tmp,mask] = imageAssert(img,mask);
unrung_SS = unring(img_tmp,mask);

tmp.img = unrung_SS;
save_untouch_nii(tmp,[foldername 'rawdata_merged_deno_derung.nii']);

%% Run co-registration

fprintf('Running co-registration...\n')

filename = 'rawdata_merged_deno_derung';

%register_image(foldername, filename)
register_image(foldername, filename)

%% Create ROI

tmp = load_untouch_nii([foldername 'rawdata_merged_deno_derung_reg.nii']);
img = tmp.img;

mask = ones(size(squeeze(img(:,:,:,1))));

m = mask;

M = mask(:);

ROI = reshape(img, [size(img,1)*size(img,2)*size(img,3) size(img,4)]);

save([foldername 'ROI'], 'ROI', 'm', 'M'); 


