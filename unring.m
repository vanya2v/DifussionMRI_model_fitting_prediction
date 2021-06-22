function unrung = unring(image,mask)
% Gibbs unringing implementation by Jonas Lynge Olesen for diffusion MRI
% data based on the algorithm presented by Kellner et al. (2015).
%
%
% input: 
% 
% image - contains MRI image data. The first dimensions must discriminate
% between pixels, while the last dimension should contain the pixelwise
% data.
%
% mask - logical array specifying which pixels to include. Can optionally
% be left unspecified in which case every pixel is included.
%
%
% output:
%
% unrung - contains corrected data.


%% adjust image dimensions and assert
dimsOld = size(image);
if nargin<3 || numel(mask)==0
    mask = true(dimsOld(1:end-1));
end
[image,mask] = imageAssert(image,mask);


%% unring image
dims = size(image);
unrung = zeros(dims);
parfor i = 1:dims(4)
    dummy = image(:,:,:,i);
    for j = 1:dims(3)
        dummy(:,:,j) = unring2D(dummy(:,:,j),mask(:,:,j)); 
    end
    unrung(:,:,:,i) = dummy;
end


%% adjust output to match input dimensions
unrung = reshape(unrung,dimsOld);


end



function unrung = unring2D(data,mask)
% Make Gx and Gy filters and apply them to original image
[Ny,Nx] = size(data);
kx = 2*pi/Nx*floor(0:Nx-1);
ky = 2*pi/Ny*floor(0:Ny-1);
[kx,ky] = meshgrid(kx,ky);
Gx = (1+cos(ky))./(2+cos(kx)+cos(ky)+eps(1));
Gy = (1+cos(kx))./(2+cos(kx)+cos(ky)+eps(1));
Gx(ky==pi & kx==pi) = 0.5;
Gy(ky==pi & kx==pi) = 0.5;

Ix = ifft2(fft2(data).*Gx);
Iy = ifft2(fft2(data).*Gy);

% Perform unringing of transformed images and combine
M = 20;
delta = 3;
maskX = any(mask,2);
maskY = any(mask,1);
Ix(maskX,:) = unring1D(Ix(maskX,:)',M,delta)';
Iy(:,maskY) = unring1D(Iy(:,maskY),M,delta);
unrung = data;
unrung(maskX,maskY) = Ix(maskX,maskY)+Iy(maskX,maskY);
if isreal(data)
    unrung = real(unrung); 
end

end



function unrung = unring1D(data,M,delta) % unrings all columns
% Calculate shifted Y vector for every value of s
[Nr,Nc] = size(data);
s = [0 1:M -(1:M)]/(2*M);
k = floor([0:Nr/2 -Nr/2+1:-1]);
if mod(Nr,2)==1, k = [k -1]; end
c = permute(fft(data,[],1),[1 3 2]);
C = bsxfun(@times,c,exp(2*pi*1i*repmat(k'*s,[1 1 Nc])/Nr));
if mod(Nr,2)==0, C(Nr/2+1,2:end,:)=0; end
shifted = ifft(C,[],1);

% Determine optimal shift for each pixel
T = zeros(size(data));
delta = delta+1;
for k = 1:Nc
    for i = 1:Nr
        xm = mod((i-delta:i-1)-1,Nr)+1;
        xp = mod((i+1:i+delta)-1,Nr)+1;
        [~,T(i,k)] = min([sum(abs(diff(shifted(xm,:,k))),1) sum(abs(diff(shifted(xp,:,k))),1)]);
    end
end
T = mod(T-1,2*M+1)+1;

% Linearly interpolate
unrung = data;
for k = 1:Nc
    for i = 1:Nr
        if s(T(i,k))>0
            unrung(i,k) = shifted(i,T(i,k),k)-(shifted(i,T(i,k),k)-shifted(mod(i-2,Nr)+1,T(i,k),k))*s(T(i,k));
        elseif s(T(i,k))<0
            unrung(i,k) = shifted(i,T(i,k),k)-(shifted(mod(i,Nr)+1,T(i,k),k)-shifted(i,T(i,k),k))*s(T(i,k));
        end
    end
end

end