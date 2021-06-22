function [] = register_image(foldername, filename)

tmp = load_untouch_nii([foldername filename '.nii']);
img = tmp.img;

mask = zeros(size(img,1), size(img,2), size(img,3));
tt = squeeze(img(:,:,:,1));
mask(tt>=5000) = 1;

tt = img;

for i =1:size(tt,4)
    
    tt(:,:,:,i) = tt(:,:,:,i).*mask;
    
end

%% Registering each block 

nbvals = 5;
nb0 = 1;
ndir = 3;

[optimizerb0, metricb0] = imregconfig('monomodal');
optimizerb0.MaximumIterations = 200;

[optimizer, metric] = imregconfig('multimodal');

optimizer.InitialRadius = 0.001;
optimizer.Epsilon = 1.5e-4;
optimizer.GrowthFactor = 1.01;
optimizer.MaximumIterations = 200;

% Registering each b0 with the first b0

tt = reshape(tt,[size(img,1), size(img,2), size(img,3), nb0 + ndir, nbvals]);

folder = [foldername 'registration/'];
mkdir(folder);

for i=1:nb0 + ndir
    
    for j=1:nbvals
        
        tmp.img = tt(:,:,:,i,j);
        tmp.hdr.dime.dim(5) = 1;
        
        if i==1
           save_untouch_nii(tmp,[folder 'B0_shell' num2str(j) '_merged_deno_derung.nii']); 
           
        else
            
           save_untouch_nii(tmp,[folder 'B_shell' num2str(j) '_Dir_' num2str(i) '_merged_deno_derung.nii']); 
           
        end
    end
end

tmp = load_untouch_nii([foldername filename '.nii']);

S0_registered = zeros(size(img,1), size(img,2), size(img,3), 5);
S0_moving = squeeze(double(tt(:,:,:,nb0,:)));
S0_fixed = double(tt(:,:,:,nb0,1));

fprintf('Registering the b0s with rigid 6dof...\n');

for jj = 1:1

for i=2:nbvals
        
    fprintf('| %2.0d%% |',((i-2)/nbvals)*100);


    for j=1:size(S0_fixed,3)
    
        for k = 1:size(S0_moving,4)
        
       tform = imregtform(S0_moving(:,:,j,k), S0_fixed(:,:,j), 'rigid', optimizerb0, metricb0);
       S0_registered(:,:,j,k) = imwarp(S0_moving(:,:,j,k),tform,'OutputView',imref2d(size(S0_fixed)));
       
        end
    end
end

S0_moving = S0_registered;

end

fprintf('\n')

tt_reg = tt;

for i = 1:nbvals
    
    if i == 1
        
        tt_reg(:,:,:,nb0,i) = S0_fixed(:,:,:);
        
    else
        
        tt_reg(:,:,:,nb0,i) = S0_registered(:,:,:,i);
        
    end
    
end

fprintf('Registering the other bs with rigid 6dof...\n');


for i=1:nbvals
    
    fprintf('| %2.0d%% |',(i./nbvals)*100);
    
    S0_moving = squeeze(double(tt_reg(:,:,:,nb0+1:nb0+ndir,i)));
    S0_fixed = squeeze(double(tt_reg(:,:,:,nb0,i)));
    
    for jj = 1:1

    for j=1:size(S0_fixed,3)
    
        for k = 1:size(S0_moving,4)
        
            tform = imregtform(S0_moving(:,:,j,k), S0_fixed(:,:,j), 'rigid', optimizer, metric);
       
            tt_reg(:,:,j,k+1,i) = imwarp(S0_moving(:,:,j,k),tform,'OutputView',imref2d(size(S0_fixed)));
       
        end
    end
    
    S0_moving = squeeze(double(tt_reg(:,:,:,nb0+1:nb0+ndir,i)));
    end
end

fprintf('\n')

fprintf('Registering all the registered images with affine 12dof...\n');

for i=1:nbvals
    
    fprintf('| %2.0d%% |',(i./nbvals)*100);
    
    S0_moving = squeeze(double(tt_reg(:,:,:,nb0+1:nb0+ndir,i)));
    S0_fixed = squeeze(double(tt_reg(:,:,:,nb0,i)));
 
    for jj = 1:1
    for j=1:size(S0_fixed,3)
    
        for k = 1:size(S0_moving,4)
        
            tform = imregtform(S0_moving(:,:,j,k), S0_fixed(:,:,j), 'Affine', optimizer, metric);%similarity &afiine
       
       
            tt_reg(:,:,j,k+1,i) = imwarp(S0_moving(:,:,j,k),tform,'OutputView',imref2d(size(S0_fixed)));
       
        end
    end
    S0_moving = squeeze(double(tt_reg(:,:,:,nb0+1:nb0+ndir,i)));
    end
end

fprintf('\n')

tmp.img = tt_reg;

save_untouch_nii(tmp,[foldername filename '_reg.nii']);

end
