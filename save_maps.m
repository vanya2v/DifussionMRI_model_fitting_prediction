addpath(genpath('MATLAB/nifti'));
addpath(genpath('MATLAB/camino'));
method = 2;
if method == 1
	database_name = 'fitdees';
elseif method == 2
	database_name = 'fixdees';
elseif method == 3
	database_name = 'fitT2s';
end


originaldata_path = '/SAN/medic/Verdict/innovate_data/dataset/INNOVATE/';
dirListing = dir(originaldata_path);

subjectname = ['INN-104-RWB/']
fprintf(['Processing subject /' subjectname ' ...\n'])

load([subjectname 'processed/ROI_DL.mat'], 'Signal', 'M');
fprintf('Preparing and saving the results...\n');

tmp = load_untouch_nii([subjectname 'processed/rawdata_merged_deno_derung_reg.nii']);
img = tmp.img;
x_dim = size(img,1);
y_dim = size(img,2);
z_dim = size(img,3);


load([subjectname 'processed/DLprediction.mat'], 'DLprediction');
mpgMean = DLprediction;
mpgMean = abs(mpgMean);

if method == 3   
     mpgMean(:,6) = mpgMean(:,6).*S0_star; % This is the predicted S0 that has to be rescaled
end





fprintf(['Subject /' subjectname ' processed in %3.0f sec.\n'])


% Convert the model parameters in VERDICT parameters 

fitted_params = struct;

if method == 1
    
    fitted_params.name = {'fvasc', 'fic', 'fees', 'R', 'Cellularity', 'Dees'};
    
    fitted_params.par(:,1) = 1-mpgMean(:,1);
    fitted_params.par(:,2) = mpgMean(:,1).*mpgMean(:,2);
    fitted_params.par(:,3) = mpgMean(:,1).*(1-mpgMean(:,2));
    
    fitted_params.par(fitted_params.par(:,1)<0,1) = 0;
    fitted_params.par(fitted_params.par(:,2)<0,2) = 0;
    fitted_params.par(fitted_params.par(:,3)<0,3) = 0;
    fitted_params.par(:,1) = fitted_params.par(:,1)./(fitted_params.par(:,1)+fitted_params.par(:,2)+fitted_params.par(:,3));
    fitted_params.par(:,2) = fitted_params.par(:,2)./(fitted_params.par(:,1)+fitted_params.par(:,2)+fitted_params.par(:,3));
    fitted_params.par(:,3) = fitted_params.par(:,3)./(fitted_params.par(:,1)+fitted_params.par(:,2)+fitted_params.par(:,3));
    
    fitted_params.par(:,4) = mpgMean(:,3);
    fitted_params.par(:,5) = fitted_params.par(:,2)./(fitted_params.par(:,4)).^3;
    fitted_params.par(:,6) = mpgMean(:,4);

elseif method == 2
    fitted_params.name = {'fvasc', 'fic', 'fees', 'R', 'Cellularity'};

        fitted_params.par(:,1) = 1-mpgMean(:,1);
    fitted_params.par(:,2) = mpgMean(:,1).*mpgMean(:,2);
    fitted_params.par(:,3) = mpgMean(:,1).*(1-mpgMean(:,2));
    
    fitted_params.par(fitted_params.par(:,1)<0,1) = 0;
    fitted_params.par(fitted_params.par(:,2)<0,2) = 0;
    fitted_params.par(fitted_params.par(:,3)<0,3) = 0;
    fitted_params.par(:,1) = fitted_params.par(:,1)./(fitted_params.par(:,1)+fitted_params.par(:,2)+fitted_params.par(:,3));
    fitted_params.par(:,2) = fitted_params.par(:,2)./(fitted_params.par(:,1)+fitted_params.par(:,2)+fitted_params.par(:,3));
    fitted_params.par(:,3) = fitted_params.par(:,3)./(fitted_params.par(:,1)+fitted_params.par(:,2)+fitted_params.par(:,3));
    
    fitted_params.par(:,4) = mpgMean(:,3);
    fitted_params.par(:,5) = fitted_params.par(:,2)./(fitted_params.par(:,4)).^3;
    
elseif method == 3
    fitted_params.name = {'fvasc', 'fic', 'fees', 'R', 'Cellularity', 'Dees', 'T2vasc_ees', 'T2ic', 'S0', 'T1'};
    
    fitted_params.par(:,1) = 1-mpgMean(:,1);
    fitted_params.par(:,2) = mpgMean(:,1).*mpgMean(:,2);
    fitted_params.par(:,3) = mpgMean(:,1).*(1-mpgMean(:,2));
    
    fitted_params.par(fitted_params.par(:,1)<0,1) = 0;
    fitted_params.par(fitted_params.par(:,2)<0,2) = 0;
    fitted_params.par(fitted_params.par(:,3)<0,3) = 0;
    fitted_params.par(:,1) = fitted_params.par(:,1)./(fitted_params.par(:,1)+fitted_params.par(:,2)+fitted_params.par(:,3));
    fitted_params.par(:,2) = fitted_params.par(:,2)./(fitted_params.par(:,1)+fitted_params.par(:,2)+fitted_params.par(:,3));
    fitted_params.par(:,3) = fitted_params.par(:,3)./(fitted_params.par(:,1)+fitted_params.par(:,2)+fitted_params.par(:,3));
    
    fitted_params.par(:,4) = mpgMean(:,3);
    fitted_params.par(:,5) = fitted_params.par(:,2)./(fitted_params.par(:,4)).^3;
    fitted_params.par(:,6) = mpgMean(:,8);
    fitted_params.par(:,7) = mpgMean(:,4);
    fitted_params.par(:,8) = mpgMean(:,5);
    fitted_params.par(:,9) = mpgMean(:,6);
    fitted_params.par(:,10) = mpgMean(:,7);

end

% Save the maps in the model parameters
for i=1:size(fitted_params.par,2)
    
    fstick = zeros(x_dim*y_dim*z_dim,1);
    
    fstick(M==1) = fitted_params.par(:,i);
    
    fstick = reshape(fstick,[x_dim y_dim z_dim]);
    tmp.img = fstick;
    tmp.hdr.dime.dim(5) = 1;
    save_untouch_nii(tmp,[subjectname 'processed/' database_name '_' fitted_params.name{i} '_DL.nii.gz']);
    
end

fprintf('DONE\n')

