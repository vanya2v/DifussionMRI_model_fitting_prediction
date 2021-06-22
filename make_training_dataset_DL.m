
%function trainingset_filename = make_training_dataset_DL(method, Nset, SNR)
addpath(genpath('../experiment1/MATLAB/nifti'))
addpath(genpath('../experiment1/MATLAB/camino'))

method=3;
Nset=1e5;
SNR=35;
tic

fprintf('Building training dataset...\n')

%% Build the database

%1 - FitDees; 2 - FixDees; 3 - FitT2s

if method==1 % Fit dees
    
    B = [1e-6 0.090 0.500 1.5 2 3];
    Delta = [23.8 23.8 31.3 43.8 34.3 38.8];
    delta = [3.9 3.9 11.4 23.9 14.4 18.9];
    protocol = make_protocol(B, Delta, delta);
    Brange = B>=0;
    
    p1 = (0.99-0.01).*rand(Nset,1)+0.01;  % ftissue
    p2 = (0.99-0.01).*rand(Nset,1)+0.01;  % fic
    p3 = (15-0.01).*rand(Nset,1)+0.01;    % Ric
    p4 = (3-1).*rand(Nset,1)+1;           % Dees
     
    f = @(p,prot) (1-p(1)).*SynthMeasAstroSticks(8E-9,prot) + p(1).*( p(2).*SynthMeasSphere([2E-9, p(3)*1E-6],prot) + (1-p(2)).*SynthMeasBall(p(4)*1E-9, prot));
    p_fit = [p1 p2 p3 p4];
    
    database_name = 'randomdire_GPD_fitdees';

elseif method == 2 % Fix dees
    
    B = [1e-6 0.090 0.500 1.5 2 3];
    Delta = [23.8 23.8 31.3 43.8 34.3 38.8];
    delta = [3.9 3.9 11.4 23.9 14.4 18.9];
    protocol = make_protocol(B, Delta, delta);
    Brange = B>=0;
    
    p1 = (0.99-0.01).*rand(Nset,1)+0.01;  % ftissue
    p2 = (0.99-0.01).*rand(Nset,1)+0.01;  % fic
    p3 = (15-0.01).*rand(Nset,1)+0.01;    % Ric

    f = @(p,prot) (1-p(1)).*SynthMeasAstroSticks(8E-9, prot) + p(1).*( p(2).*SynthMeasSphere([2E-9, p(3)*1E-6],prot) + (1-p(2)).*SynthMeasBall(2E-9, prot));
    p_fit = [p1 p2 p3];
    
    database_name = 'randomdire_GPD_fixdees';
    
elseif method == 3 % Fit T2s
    
    B = [1e-6 0.090 1e-6 0.500 1e-6 1.5 1e-6 2 1e-6 3];
    Delta = [23.8 23.8 31.3 31.3 43.8 43.8 34.3 34.3 38.8 38.8];
    delta = [3.9 3.9 11.4 11.4 23.9 23.9 14.4 14.4 18.9 18.9];
    TE = [50 50 65 65 90 90 71 71 80 80]';
    TR = [2482 2482 2482 2482 2482 2482 3945 3945 3349 3349]';
    protocol = make_protocol(B, Delta, delta);
    Brange = B>=0;
    
    p1 = (0.99-0.01).*rand(Nset,1)+0.01; % ftissue
    p2 = (0.99-0.01).*rand(Nset,1)+0.01; % fic
    p3 = (15-0.01).*rand(Nset,1)+0.01;   % Ric
    p4 = (800-150).*rand(Nset,1)+150;    % T2vasc/ees
    p5 = (150-1).*rand(Nset,1)+1;        % T2ic
    p6 = (1-0.1).*rand(Nset,1)+0.1;      % S0
    p7 = (4e3-10).*rand(Nset,1)+10;      % T1
    p8 = (3-0.5).*rand(Nset,1)+0.5;      % Dees


    f = @(p,prot) p(6).*(1-exp(-TR./p(7))).*( (1-p(1)).*exp(-TE./p(4)).*SynthMeasAstroSticks(8E-9,prot) + p(1).*(p(2).*exp(-TE./p(5)).*SynthMeasSphere([2E-9, p(3)*1E-6],prot) + (1-p(2)).*exp(-TE./p(4)).*SynthMeasBall(p(8)*1E-9, prot)) );
    p_fit = [p1 p2 p3 p4 p5 p6 p7 p8];

    
    database_name = 'fitT2s';
end

database = zeros(size(p_fit,1), length(B(Brange)));

parfor i=1:size(p_fit,1)
    
    database(i,:) = f(p_fit(i,:),protocol);
    
end

params = p_fit;

database_train = database;
params_train = params;

% Add experimental SNR for training

% Gaussian noise
%database_train_noisy = sqrt((database_train).^2 + (1./SNR(1).*randn(size(database_train))).^2);

% Rician noise
%database_train_noisy = sqrt((database_train + 1./SNR(1).*randn(size(database_train))).^2 + (1./SNR(1).*randn(size(database_train))).^2);

params_train_noisy = params_train;

 %for i = 2:numel(SNR)
     
  %  database_train_noisy = [database_train_noisy; sqrt((database_train + 1./SNR(i).*randn(size(database_train))).^2 + (1./SNR(i).*randn(size(database_train))).^2)];
 
   %  params_train_noisy = [params_train_noisy; params_train];
     
 %end

% Multiple random Rician noise in the range [25 100]

%SNR = (100 - 25).*rand(size(database_train)) + 25;
%database_train_noisy = sqrt((database_train + 1./SNR.*randn(size(database_train))).^2 + (1./SNR.*randn(size(database_train))).^2);

trainingset_filename = ['database_train_DL_' database_name '_noisefree.mat'];

%save(trainingset_filename, 'database_train_noisy', 'params_train_noisy');
save(trainingset_filename, 'database_train', 'params_train');
tt = toc;

fprintf('Training set built in %5.0f sec.\n',tt)





 
