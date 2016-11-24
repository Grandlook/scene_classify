% Starter code prepared by James Hays for CS 143, Brown University

%This feature representation is described in the handout, lecture
%materials, and Szeliski chapter 14.

function image_feats = get_bags_of_sifts(image_paths)
% image_paths is an N x 1 cell array of strings where each string is an
% image path on the file system.

% This function assumes that 'vocab.mat' exists and contains an N x 128
% matrix 'vocab' where each row is a kmeans centroid or visual word. This
% matrix is saved to disk rather than passed in a parameter to avoid
% recomputing the vocabulary every time at significant expense.

% image_feats is an N x d matrix, where d is the dimensionality of the
% feature representation. In this case, d will equal the number of clusters
% or equivalently the number of entries in each image's histogram.

% You will want to construct SIFT features here in the same way you
% did in build_vocabulary.m (except for possibly changing the sampling
% rate) and then assign each local feature to its nearest cluster center
% and build a histogram indicating how many times each cluster was used.
% Don't forget to normalize the histogram, or else a larger image with more
% SIFT features will look very different from a smaller version of the same
% image.

%{
Useful functions:
[locations, SIFT_features] = vl_dsift(img) 
 http://www.vlfeat.org/matlab/vl_dsift.html
 locations is a 2 x n list list of locations, which can be used for extra
  credit if you are constructing a "spatial pyramid".
 SIFT_features is a 128 x N matrix of SIFT features
  note: there are step, bin size, and smoothing parameters you can
  manipulate for vl_dsift(). We recommend debugging with the 'fast'
  parameter. This approximate version of SIFT is about 20 times faster to
  compute. Also, be sure not to use the default value of step size. It will
  be very slow and you'll see relatively little performance gain from
  extremely dense sampling. You are welcome to use your own SIFT feature
  code! It will probably be slower, though.

D = vl_alldist2(X,Y) 
   http://www.vlfeat.org/matlab/vl_alldist2.html
    returns the pairwise distance matrix D of the columns of X and Y. 
    D(i,j) = sum (X(:,i) - Y(:,j)).^2
    Note that vl_feat represents points as columns vs this code (and Matlab
    in general) represents points as rows. So you probably want to use the
    transpose operator '  You can use this to figure out the closest
    cluster center for every SIFT feature. You could easily code this
    yourself, but vl_alldist2 tends to be much faster.

Or:

For speed, you might want to play with a KD-tree algorithm (we found it
reduced computation time modestly.) vl_feat includes functions for building
and using KD-trees.
 http://www.vlfeat.org/matlab/vl_kdtreebuild.html

%}

load('vocab.mat')
vocab_size = size(vocab, 2);
image_feats=[];
num=length(image_paths);
for i=1:num
    img=imread(char(image_paths(i)));
    im_single=im2single(img);
    flag=0;
    if flag==1 
        hist=zeros(1,vocab_size);
        %%提取SIFT特征
        [~, SIFT_features] = vl_dsift(im_single);
        %将采样取特征
       %sample_feature=floor(linspace(1,size(SIFT_features,2),2000));
       %随机取特征
%         sample_feature=floor(rand(1,4000).*size(SIFT_features,2));
%         sample_feature(sample_feature==0)=1;
%  for j=1:size(sample_feature,2)
        for j=1:size(SIFT_features,2)
            
%             cursor_feature=SIFT_features(:,sample_feature(j));
        cursor_feature=SIFT_features(:,j);
            cursor_feature=im2single(cursor_feature);
            dis=vl_alldist2(cursor_feature,vocab);
            [~,ind]=min(dis);
            hist(1,ind)=hist(1,ind)+1;
        end;
        %%%归一化
        hist=hist./sum(hist);
        image_feats=[image_feats;hist];
    else
        %%下面是实现的beyondOf词袋模型的算法，采取3层金字塔结构。
        %%lo start.
        hist=zeros(1,vocab_size);
        %%提取SIFT特征
        [~, SIFT_features] = vl_dsift(im_single);
        %将采样取特征
       %sample_feature=floor(linspace(1,size(SIFT_features,2),2000));
       %随机取特征
        sample_feature=floor(rand(1,4000).*size(SIFT_features,2));
        sample_feature(sample_feature==0)=1;

        for j=1:size(sample_feature,2)
            cursor_feature=SIFT_features(:,sample_feature(j));
            cursor_feature=im2single(cursor_feature);
            dis=vl_alldist2(cursor_feature,vocab);
            [~,ind]=min(dis);
            hist(1,ind)=hist(1,ind)+1;
        end;
        hist=hist/2;
        %%%归一化
        %hist=hist./sum(hist);
        %%l1start.
        for i=1:4
            cuhist=zeros(1,vocab_size);
            [x_len,y_len]=size(im_single);
            cuim_single=[];
            if i==1
                cuim_single=im_single(1:floor(x_len/2),1:floor(y_len/2));
            elseif i==2
                cuim_single=im_single(1:floor(x_len/2),(floor(y_len/2)+1):y_len);
            elseif i==3
                cuim_single=im_single((floor(x_len/2)+1):x_len,1:floor(y_len/2));
            else
                cuim_single=im_single((floor(x_len/2)+1):x_len,(floor(y_len/2)+1):y_len);
            end
            %%提取SIFT特征
            [~, SIFT_features] = vl_dsift(cuim_single);
            %将采样取特征
           %sample_feature=floor(linspace(1,size(SIFT_features,2),2000));
           %随机取特征
            sample_feature=floor(rand(1,1000).*size(SIFT_features,2));
            sample_feature(sample_feature==0)=1;

            for j=1:size(sample_feature,2)
                cursor_feature=SIFT_features(:,sample_feature(j));
                cursor_feature=im2single(cursor_feature);
                dis=vl_alldist2(cursor_feature,vocab);
                [~,ind]=min(dis);
                cuhist(1,ind)=hist(1,ind)+1;
            end;
            %%%归一化
            %cuhist=cuhist./sum(cuhist);
            cuhist=cuhist/2;
            hist=[hist,cuhist];
        end
         %%l2start.
        for i=1:16
            cuhist=zeros(1,vocab_size);
            [x_len,y_len]=size(im_single);
            cuim_single=[];
            len_y=floor(y_len/4);
            len_x=floor(x_len/4);
            if ceil(i/4)==1
                
                if mod(i,4)==0
                    y_window=y_len;
                else
                    y_window=len_y*i;
                end
                cuim_single=im_single(1:floor(x_len/4),y_window-len_y+1:y_window);
                
            elseif ceil(i/4)==2
                
                if mod(i,4)==0
                    y_window=y_len;
                else
                    y_window=len_y*mod(i,4);
                end
                
                cuim_single=im_single(len_x+1:len_x*2,y_window-len_y+1:y_window);
            elseif ceil(i/4)==3
                if mod(i,4)==0
                    y_window=y_len;
                else
                    y_window=len_y*mod(i,4);
                end
                cuim_single=im_single(len_x*2+1:len_x*3,y_window-len_y+1:y_window);
            else
                if mod(i,4)==0
                    y_window=y_len;
                else
                    y_window=len_y*mod(i,4);
                end
                cuim_single=im_single(x_len-len_x+1:x_len,y_window-len_y+1:y_window);
            end
            %%提取SIFT特征
            [~, SIFT_features] = vl_dsift(cuim_single);
            %将采样取特征
           %sample_feature=floor(linspace(1,size(SIFT_features,2),2000));
           %随机取特征
            sample_feature=floor(rand(1,400).*size(SIFT_features,2));
            sample_feature(sample_feature==0)=1;
            for j=1:size(sample_feature,2)
                cursor_feature=SIFT_features(:,sample_feature(j));
                cursor_feature=im2single(cursor_feature);
                dis=vl_alldist2(cursor_feature,vocab);
                [~,ind]=min(dis);
                cuhist(1,ind)=hist(1,ind)+1;
            end;
            %%%归一化
            %cuhist=cuhist./sum(cuhist);
            cuhist=cuhist/sqrt(2);
            hist=[hist,cuhist];
        end
        %%%归一化
        hist=hist./sum(hist);
        image_feats=[image_feats;hist];
    end
    
end;



