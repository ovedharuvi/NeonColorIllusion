addpath('C:\Users\Hadar\Google Drive\PoissonEdiitng20151105');
addpath('C:\Users\Hadar\Google Drive\Diffusion-color');

imagepath = 'wc_op.png'
%imagepath = 'watercolor2.png'
%imagepath = 'wc_yb_ver1.png'
%imagepath = 'Realcolor.png';
%imagepath= 'non_a_watercolor_4_darker.png'
imagepath= 'non_watercolor_4_ic.png'
%imagepath = 'bg_watercolred_cyan.png'
%imagepath = 'red_cyan_dark_bg.png'
%imagepath = 'C:\Users\Hadar\Google Drive\illusions\cornsweet\17.png';

%imagepath = 'C:\Users\Hadar\Google Drive\illusions\cornsweet\csbox.png'
%imagepath = 'cornsweet.png'
%imagepath = 'wc_pg_bg.png'
%imagepath = 'wc_background_4.png'
%imagepath = 'Watercolor-illusion.jpg'
%imagepath = 'red-strawberries.jpg'
%imagepath='C:\Users\Hadar\Google Drive\illusions\210088.jpg'
I1 = im2double(imresize(imread(imagepath),0.7));
%I= repmat(I(:,:,1),1,1,3);
%I = I(1:200,:,:);
%I(:,:,2) = zeros(size(I(:,:,2)));
%I(:,:,3) = ones(size(I(:,:,3)));
%I = cat(1,I1, im2double(imresize(imread('wc_rm.png'),[size(I1,1) size(I1,2)])),im2double(imresize(imread('wc_cg1.png'),[size(I1,1) size(I1,2)])));
%I = cat(1,I1, im2double(imresize(imread('wc_rc.png'),[size(I1,1) size(I1,2)])),im2double(imresize(imread('wc_ac.png'),[size(I1,1) size(I1,2)])));

I = cat(1,I1, im2double(imresize(imread('non_a_watercolor_4_darker.png'),[size(I1,1) size(I1,2)])));



minS = min(size(I,1),size(I,2));


N =2
Iop  = ConvertFormRGBToOpponent1( I);



L= 0.5*[ 0,-1,0 ;-1,4,-1;0,-1,0];





clear R;
W1 = 0;
Wv1 = 0;
[BlurredPyramidGray,~, padR, padC] = GenerateBlurredPyramid(ConvertFormRGBToOpponent(repmat(rgb2gray(I),1,1,3)),N);
[BlurredPyramidWhite,~, padR, padC] = GenerateBlurredPyramid(ConvertFormRGBToOpponent(ones(size(I))),N);
[BlurredPyramid,~, padR, padC] = GenerateBlurredPyramid((Iop(:,:,1)),N);
[h,w] = size(BlurredPyramid{1})
W1 = zeros(h,w,3);
for c = 1:3
    [BlurredPyramid,~, padR, padC] = GenerateBlurredPyramid((Iop(:,:,c)),N);
    %[BlurredPyramidGray,~, padR, padC] = GenerateBlurredPyramid((Iop(:,:,3)),N);
    W = abs(imfilter(abs(BlurredPyramid{N+1}).*(1-0),L,'replicate'));
    %Wv = abs(imfilter(abs(BlurredPyramid{N+1}).*(1-0),kv1,'replicate'));
    % Whs = abs(imfilter((BlurredPyramid{N+1}),kh1,'replicate'));
    %Wvs = abs(imfilter((BlurredPyramid{N+1}),kv1,'replicate'));
     if(max(W(:))>0)
            W = W./max(W(:));
     end;
        
%   if(max(Wh(:))>0)
%             Wh = Wh./max(Wh(:));
%   end;
%   if(max(Wv(:))>0)
%             Wv = Wv./max(Wv(:));
%   end;
%     
    for i = N:-1:1

        
        Gn = abs(imfilter(abs(BlurredPyramid{i}).*(1-0),L,'replicate'));
        %Gvn = abs(imfilter(abs(BlurredPyramid{i}).*(1-0),kv1,'replicate'));
%          if(max(Ghn(:))>0)
%             Ghn = Ghn./max(Ghn(:));
%          end;
%          if(max(Gvn(:))>0)
%                    Gvn = Gvn./max(Gvn(:));
%          end;
        W = max(my_impyramid(W,'expand') , Gn);
        %Wv = max(my_impyramid(Wv,'expand') , Gvn);
        %Whs = (my_impyramid(Whs,'expand') + Ghn);
        %Wvs = (my_impyramid(Wvs,'expand') + Gvn);
       % Wh = Whm./(Whs+0.1);
       % Wv = Wvm./(Wvs+0.1);
        if(max(W(:))>0)
            W = W./max(W(:));
        end;
        

        %R = R./max(R(:));
    end
%    [Gh ,Gv] = imgrad(BlurredPyramid{1});
%     if (i < 2)
%         Wh = 0.7.*Wh;
%         Wv = 0.7.*Wv;
%     end
   W1(:,:,c) = W;
   % Wv1 = Wv+Wv1;
end
W = max(W1,[],3);

%W = sum(W1,3)./3;
for c = 1:3
     [BlurredPyramid,~, padR, padC] = GenerateBlurredPyramid((Iop(:,:,c)),N);
     [Gh ,Gv] = imgrad(BlurredPyramid{1});   
     
     %BlurredPyramid1 = 1.2*imfilter( BlurredPyramid{1}, fspecial('gaussian',3,0.1),'replicate')-0.2*imfilter( BlurredPyramid{1}, fspecial('gaussian',7,2),'replicate');
     
     R(:,:,c) = poisson_solver_function(1*(0.5*(W)+1.0).*Gh,1*(0.5*(W)+1.0).*Gv,(BlurredPyramid{1}));
    
      %  R(:,:,c) = poisson_solver_function(1*(0.5*(W)+1.0).*Gh,1*(0.5*(W)+1.0).*Gv,R(:,:,c));
end
    

%imtool( R./max(R(:)));
%imtool( W1,[]);
Rrgb = ConvertFormOpponentToRgb1( R );
m = max(Rrgb(:))-1;
%Rrgb = Rrgb - m;
% Rrgb(:,:,1) = Rrgb(:,:,1) - min(min(Rrgb(:,:,1)));
% Rrgb(:,:,1) = Rrgb(:,:,1)./max(max(Rrgb(:,:,1)));
% Rrgb(:,:,2) = Rrgb(:,:,2) - min(min(Rrgb(:,:,2)));
% Rrgb(:,:,2) = Rrgb(:,:,2)./max(max(Rrgb(:,:,2)));
% Rrgb(:,:,3) = Rrgb(:,:,3) - min(min(Rrgb(:,:,3)));
% Rrgb(:,:,3) = Rrgb(:,:,3)./max(max(Rrgb(:,:,3)));
FinalImage = Rrgb(1:end-padR,1:end-padC,:);

% s = 51;
% B1 = ordfilt2(FinalImage(:,:,1),round(s^2/2),ones(s,s));
% B2 = ordfilt2(FinalImage(:,:,2),round(s^2/2),ones(s,s));
% B3 = ordfilt2(FinalImage(:,:,3),round(s^2/2),ones(s,s));
% B = cat(3,B1,B2,B3);
% M = max(B,[],3);
%  imtool(FinalImage./M);
%imtool( FinalImage);
imtool( FinalImage./max(FinalImage(:)));
%imtool(0.5*I+0.5* FinalImage./max(FinalImage(:)));
[BlurredPyramid,~, padR, padC] = GenerateBlurredPyramid((Iop),N);
Io = BlurredPyramid{1};

E = (Io(:,:,1)==0).*( Io(:,:,2)==0);
%imtool( (E).*Rrgb./max(Rrgb(:)));
RI = I(145,:,1);
GI = I(145,:,2);
BI = I(145,:,3);

RR = FinalImage(145,:,1);
GR = FinalImage(145,:,2);
BR = FinalImage(145,:,3);
x = 1:1:length(I(145,:,1));

figure;plot(x,RR,x,RI)
xlabel('Position[pixels]') % x-axis label
ylabel('Normalized intensity') %

