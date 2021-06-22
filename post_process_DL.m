
addpath(genpath('MATLAB/nifti'));
addpath(genpath('MATLAB/camino'));
method = 2;


originaldata_path = '/SAN/medic/Verdict/innovate_data/dataset/INNOVATE/';
dirListing = dir(originaldata_path);

subjectname = ['INN-104-RWB/']
fprintf(['Processing subject /' subjectname ' ...\n'])

tic

%% Create ROI

fprintf('Preparing the data for DL fit...\n');

foldername = [subjectname 'processed/']; 
% Uncomment in case you have a mask and prfer to constraint the analysis to
% only the voxels within the mask

%tmp = load_untouch_nii([subjectname 'processed/mask.nii.gz']);
%mask_img = tmp.img;
%mask = reshape(mask_img, [size(mask_img,1)*size(mask_img,2)*size(mask_img,3) size(mask_img,4)]);
%m = mask;
%M = m(:);

%fprintf(['Mask used' subjectname '\n'])
%tmp = load_untouch_nii([foldername 'rawdata_merged_deno_derung_reg.nii']);
%img = tmp.img;

fprintf(['WARNING: Found no specific mask for patient ' subjectname ' so processing the whole image\n'])
tmp = load_untouch_nii([foldername 'rawdata_merged_deno_derung_reg.nii']);
img = tmp.img;
mask = ones(size(squeeze(img(:,:,:,1))));
m = mask;
M = m(:);

% Apply a 2D gaussian smoothing slice per slice 

for i=1:size(img,4)
    for j=1:size(img,3)
        
        img(:,:,j,i) = imgaussfilt(img(:,:,j,i), 0.7);
        
    end
end

ROI = reshape(img, [size(img,1)*size(img,2)*size(img,3) size(img,4)]);
ROI = ROI(M==1,:);

%% Prepare data for fitting

nbvals = 5;

ROItmp = zeros(size(ROI,1), nbvals+1);
ROI_b0s_tmp = zeros(size(ROI,1), nbvals*2);

ROItmp(:,1) = 1.0;

k = 1;
l = 1;

for i = 1:nbvals
    
    ROItmp(:,i+1) = nanmean(ROI(:,k+1:k+3)./ROI(:,k),2);
    ROI_b0s_tmp(:,l) = ROI(:,k);
    ROI_b0s_tmp(:,l+1) = nanmean(ROI(:,k+1:k+3),2);
    k = k+4;
    l = l+2;
    
end

ROItmp(isnan(ROItmp)) = 0;
ROI_b0s_tmp(isnan(ROI_b0s_tmp)) = 0;

ROItmp(ROItmp>1) = 1;
ROItmp(ROItmp<0) = 1e-6;

ROI_b0s_tmp(ROI_b0s_tmp<0) = 1e-6;

if method == 1
    
    B = [1e-6 0.090 0.500 1.5 2 3];
    Delta = [23.8 23.8 31.3 43.8 34.3 38.8];
    delta = [3.9 3.9 11.4 23.9 14.4 18.9];
    protocol = make_protocol(B, Delta, delta);
    f = @(p,prot) (1-p(1)).*SynthMeasAstroSticks(8E-9,prot) + p(1).*( p(2).*SynthMeasSphere([2E-9, p(3)*1E-6],prot) + (1-p(2)).*SynthMeasBall(p(4)*1E-9, prot));
    
    database_name = 'fitdees';
    
    Signal = ROItmp;
    
    
elseif method == 2
    
    B = [1e-6 0.090 0.500 1.5 2 3];
    Delta = [23.8 23.8 31.3 43.8 34.3 38.8];
    delta = [3.9 3.9 11.4 23.9 14.4 18.9];
    protocol = make_protocol(B, Delta, delta);
    f = @(p,prot) (1-p(1)).*SynthMeasAstroSticks(8E-9, prot) + p(1).*( p(2).*SynthMeasSphere([2E-9, p(3)*1E-6],prot) + (1-p(2)).*SynthMeasBall(2E-9, prot));
    
    database_name = 'fixdees';
    
    Signal = ROItmp;
        
    
elseif method == 3
    
    B = [1e-6 0.090 1e-6 0.500 1e-6 1.5 1e-6 2 1e-6 3];
    Delta = [23.8 23.8 31.3 31.3 43.8 43.8 34.3 34.3 38.8 38.8];
    delta = [3.9 3.9 11.4 11.4 23.9 23.9 14.4 14.4 18.9 18.9];
    TE = [50 50 65 65 90 90 71 71 80 80]';
    TR = [2482 2482 2482 2482 2482 2482 3945 3945 3349 3349]';
    protocol = make_protocol(B, Delta, delta);
    f = @(p,prot) p(6).*(1-exp(-TR./p(7))).*( (1-p(1)).*exp(-TE./p(4)).*SynthMeasAstroSticks(8E-9,prot) + p(1).*(p(2).*exp(-TE./p(5)).*SynthMeasSphere([2E-9, p(3)*1E-6],prot) + (1-p(2)).*exp(-TE./p(4)).*SynthMeasBall(p(8)*1E-9, prot)) );

    
    database_name = 'fitT2s';
        
    Signal = ROI_b0s_tmp; 

    S0_star = max(ROI_b0s_tmp,[],2).*2; % Because in relaxed-VERDICT we need to also estimate the S0, here we divide each voxel for twice the max of the signal in that voxel and train for S0 in the range [0 1]. The signal at b = 0, TE = 0 and TR = inf can be <~ 2*max
 
    Signal = Signal./S0_star;
    
    Signal(isnan(Signal)) = 0;
    Signal(isinf(Signal)) = 0;
        
end

save([subjectname 'processed/ROI_DL.mat'], 'm', 'M', 'Signal')


