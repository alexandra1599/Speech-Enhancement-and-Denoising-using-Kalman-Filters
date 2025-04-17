%First you should download the dataset from NOIZEUS − http://ecs.utdallas.edu/loizou/speech/noizeus/
%This will contain speech files for 3 types of noise : white, train and
%babble
%each type of noise will have 3 subfolders containing speech signals
%corrupted by the specified noise at different SNRs (0,5,10 dB)

% EXAMPLE on how to run the algorithm
%run this same line for different typenoise (white,babble,train) for all
%the SNRs, the first argument is the name of the file, the second is the
%noise type and the third is the LPC order

tuned_kalman_filter('sp02', 'train_noise', 50);

function [S] = tuned_kalman_filter(filename, typenoise, order_LPC)
%Applies tuned Kalman filter with order estimation on noisy speech.
%noiseType - white, train or babble

ppath = fileparts(pwd);
%download from http://ecs.utdallas.edu/loizou/speech/software.htm

addpath(strcat(ppath,'/Users/alexandramikhael/Downloads/composite/'));
%this folder contains clean and corrupted .wav speech files 
    
%read a clean audio signal
[z,fs] = audioread('/Users/alexandramikhael/Desktop/MASTERS/Spring 2023/ECE 251B/PROJECT/clean/sp01.wav');
%read a corrupted audio signal
if(strcmp(typenoise, 'white') == 1)
   wn = audioread('/Users/alexandramikhael/Downloads/white_noise.wav');
   [noise,snr] = makeSNR(z, wn, SNR(p));
   y = noise + z;   
   fs = 48000;
elseif(strcmp(typenoise, 'babble') == 1)
   wn = audioread('/Users/alexandramikhael/Downloads/babble_noise.wav');
   [noise,snr] = makeSNR(z, wn, SNR(p));
   y = noise + z;
   fs = 48000;
else
   fs = 48000;
   [y,~] = audioread('/Users/alexandramikhael/Desktop/MASTERS/Spring 2023/ECE 251B/PROJECT/airport_noise/sp02_airport_sn5.wav');

end
    y=y';
    z=z';
    
    %dividing into 80ms frames with 10ms overlap
    start=1;
    l = 0.08*fs;
    overlap = 0.01*fs;
    totframe = ceil(length(y)/(l-overlap));
    speech_segment=zeros(totframe,l);
    z_seg=zeros(totframe,l);
    
    for i=1:totframe-1
        speech_segment(i,1:l)=y(1,start:start+l-1);
        z_seg(i,1:l)=z(1,start:start+l-1);
        start=(l-overlap)*i+1;
    end
    
    speech_segment(totframe,1:length(y)-start+1)=y(start:length(y));
    z_seg(totframe,1:length(z)-start+1)=z(start:length(z));
    clean_speech=zeros(totframe,l);
    S=zeros(1,length(y));
    
    %determine order of the LPC
    if strcmp(order_LPC,'fixed') == 1
        order_LPC = ones(totframe,1).*15;
    else
        order_LPC = findOrder(speech_segment);
    end
    
    %calculate measurement noise variance R
    R = measurementNoiseNew(speech_segment, fs);
    
    %initializing the arrays
    J1=zeros(1,10);
    J2=zeros(1,10);
    nq=zeros(1,10);
    u=1;
    
    for m = 0:6
        
        snr_before=0;
        snr_after=0;
        Q_arr=zeros(1,totframe);
        
        for i=1:totframe
            %initializing
            X = y(1:order_LPC(i))';
            P = zeros(l,order_LPC(i),order_LPC(i));
            t1 = zeros(order_LPC(i),order_LPC(i));
            H = [zeros(1,order_LPC(i)-1),1];
            G = H';
            
            %first iteration of Kalman filter
            [A,Q1] = lpc(speech_segment(i,:),order_LPC(i));
            temp = eye(order_LPC(i));
            phi_kalman = [temp(2:order_LPC(i),:);-fliplr(A(2:end))];
            
            if(i == 1)
                P(1,:,:)= R*eye(order_LPC(i));
            else
                P(1,:,:) = Y(:,:);
            end
            
            %calculating optimum value of process noise variance, Q
            q=1;

            %functions for J1, J2 and nq are taken by literature (view
            %paper)
            for n=-5:4
                Q0=(10^n)*Q1;
                t1(:,:)=P(1,:,:);
                Ak=H*(phi_kalman*t1*phi_kalman')*H';
                Bk=H*Q0*H';
                J1(q)=R/(Ak+Bk+R);
                J2(q)=Bk/(Ak+Bk);
                nq(q)=log10(Bk);
                q=q+1;
            end
            
            %interpolate nq, J1 and J2 to increase resolution, and to get more
            %accurate approximation of Q
            nqi = -5:0.25:4;
            J2i = interp1(nq,J2,nqi);
            J1i = interp1(nq,J1,nqi);
            [~,Jc]=intersections(nqi,J1i,nqi,J2i);

            %formlas taken from literature (see paper)
            
            if m < 3
                J2_desired = (0.25*(m+1)*(Jc-min(J2i))) + min(J2i);
            else
                J2_desired = (0.25*(m-3)*(max(J2i)-Jc))+ Jc;
            end
            
            [~, index] = min(abs(J2i - J2_desired));
            Q = 10^(nqi(index));
            
            %plot J1,J2 for voiced frame
            if(i == 4)
                figure();
                plot(nqi,J1i,'-r+');
                hold on;
                grid on;
                plot(nqi,J2i,'-r*');
                hold on;
                grid on;
                scatter(nqi(index),J2i(index),'k');
            end
            
            %plot J1,J2 for silent frame
            if(i == totframe - 1)
                figure();
                plot(nqi,J1i,'-b+');
                hold on;
                grid on;
                plot(nqi,J2i,'-b*');
                hold on;
                grid on;
                scatter(nqi(index),J2i(index),'k');
                xlabel('nq','FontSize',18);
                ylabel('JI,J2','FontSize',16);
                axis([min(nqi)-2, max(nqi), 0, 1]);
                hold off;
                legend('J_1','J_2');
            end
            
            Q_arr(u)=Q;
            u=u+1;
            
            for j=1:length(speech_segment(i,:))
                X_k=phi_kalman*X;
                t1(:,:)=P(j,:,:);
                P_k=(phi_kalman*t1*phi_kalman')+(G*Q*G');
                K=(P_k*H')*(inv(H*P_k*H'+R));
                t1=(eye(order_LPC(i))-K*H)*P_k;
                P(j+1,:,:)=t1(:,:);
                e=speech_segment(i,j)-(H*X_k);
                X=X_k+K*e;
                clean_speech(i,j)=X(end);
                
            end
            
            %adjust a posteriori error covariance matrix dimensions
            if(i< totframe)
                t2 = zeros(order_LPC(i),order_LPC(i));
                t2(:,:) = P(j-1,:,:);
                Y = adjustDimensions(t2, order_LPC(i+1));
            end
            
            %second iteration of Kalman filter with lpc calculated from
            %cleaned speech
            [A,Q]=lpc(clean_speech(i,:),order_LPC(i));
            phi_kalman=[temp(2:order_LPC(i),:);-fliplr(A(2:end))];
            X=clean_speech(i,1:order_LPC(i))';
            
            if i==1
                P0=R*eye(order_LPC(i));
            else
                P0 = Z(:,:);
            end
            
            for j=1:length(speech_segment(i,:))
                X_k=phi_kalman*X;
                P_k=(phi_kalman*P0*phi_kalman')+(G*Q*G');
                K=(P_k*H')*(inv(H*P_k*H'+R));
                P0=(eye(order_LPC(i))-K*H)*P_k;
                e=speech_segment(i,j)-(H*X_k);
                X=X_k+K*e;
                clean_speech(i,j)=X(end);
            end
            
            if(i< totframe)
                Z = adjustDimensions(P0, order_LPC(i+1));
            end
            
            %calculate Segmental SNR
            snr_before=snr_before+log10(rms(z_seg(i,:))/rms(z_seg(i,:)-speech_segment(i,:)));
            snr_after=snr_after+log10(rms(z_seg(i,:))/rms(z_seg(i,:)-clean_speech(i,:)));    
        end
        
        %overlap add
        S(1:l)=clean_speech(1,1:l);
        start=l+1;
        for i=2:totframe-1
            S(start:start+(l-overlap))=clean_speech(i,overlap:end);
            start=start+l-overlap-1;
        end
        S(start:length(y))=clean_speech(totframe,1:(length(y)-start)+1);
        S=S(1:length(y));
        
        %normalizing
        z=z./abs(1.2*max(z));
        y=y./abs(1.2*max(y));
        S=S./abs(1.2*max(S));
        
        %qualitative measure of noise removed
        figure();ylabel('Normalised amplitude');xlabel('Time in seconds');
        subplot(3,1,1);plot((1:length(z))/fs,z);title('original speech');axis([0, length(z)/fs, -1, 1]);
        subplot(3,1,2);plot((1:length(y))/fs,y,'k');title('corrupted speech');axis([0, length(y)/fs, -1, 1]);
        subplot(3,1,3);plot((1:length(S))/fs,S,'r');title('cleaned speech');axis([0, length(S)/fs, -1, 1]);
        
        figure();
        subplot(3,1,1);spectrogram(z,64,16,1024,fs,'yaxis');
        title('Spectrum of clean speech');
        subplot(3,1,2);spectrogram(y,64,16,1024,fs,'yaxis');
        title('Spectrum of noisy speech');
        subplot(3,1,3);spectrogram(S,64,16,1024,fs,'yaxis');
        title('Spectrum of cleaned speech after processing');
        
        %quantitative measure of noise removed - Seg SNR
        disp('The segmental snr before processing is :');
        snr_before = 20*snr_before/totframe
        disp('The segmental snr after processing is :');
        snr_after = 20*snr_after/totframe
        
  
end
fclose('all');
end


function [x0,y0,iout,jout] = intersections(x1,y1,x2,y2,robust)
%INTERSECTIONS Intersections of curves.
%   Computes the (x,y) locations where two curves intersect.  The curves
%   can be broken with NaNs or have vertical segments.
%
% Example:
%   [X0,Y0] = intersections(X1,Y1,X2,Y2,ROBUST);
%
% where X1 and Y1 are equal-length vectors of at least two points and
% represent curve 1.  Similarly, X2 and Y2 represent curve 2.
% X0 and Y0 are column vectors containing the points at which the two
% curves intersect.
%
% ROBUST (optional) set to 1 or true means to use a slight variation of the
% algorithm that might return duplicates of some intersection points, and
% then remove those duplicates.  The default is true, but since the
% algorithm is slightly slower you can set it to false if you know that
% your curves don't intersect at any segment boundaries.  Also, the robust
% version properly handles parallel and overlapping segments.
%
% The algorithm can return two additional vectors that indicate which
% segment pairs contain intersections and where they are:
%
%   [X0,Y0,I,J] = intersections(X1,Y1,X2,Y2,ROBUST);
%
% For each element of the vector I, I(k) = (segment number of (X1,Y1)) +
% (how far along this segment the intersection is).  For example, if I(k) =
% 45.25 then the intersection lies a quarter of the way between the line
% segment connecting (X1(45),Y1(45)) and (X1(46),Y1(46)).  Similarly for
% the vector J and the segments in (X2,Y2).
%
% You can also get intersections of a curve with itself.  Simply pass in
% only one curve, i.e.,
%
%   [X0,Y0] = intersections(X1,Y1,ROBUST);
%
% where, as before, ROBUST is optional.

% Version: 1.12, 27 January 2010
% Author:  Douglas M. Schwarz
% Email:   dmschwarz=ieee*org, dmschwarz=urgrad*rochester*edu
% Real_email = regexprep(Email,{'=','*'},{'@','.'})


% Theory of operation:
%
% Given two line segments, L1 and L2,
%
%   L1 endpoints:  (x1(1),y1(1)) and (x1(2),y1(2))
%   L2 endpoints:  (x2(1),y2(1)) and (x2(2),y2(2))
%
% we can write four equations with four unknowns and then solve them.  The
% four unknowns are t1, t2, x0 and y0, where (x0,y0) is the intersection of
% L1 and L2, t1 is the distance from the starting point of L1 to the
% intersection relative to the length of L1 and t2 is the distance from the
% starting point of L2 to the intersection relative to the length of L2.
%
% So, the four equations are
%
%    (x1(2) - x1(1))*t1 = x0 - x1(1)
%    (x2(2) - x2(1))*t2 = x0 - x2(1)
%    (y1(2) - y1(1))*t1 = y0 - y1(1)
%    (y2(2) - y2(1))*t2 = y0 - y2(1)
%
% Rearranging and writing in matrix form,
%
%  [x1(2)-x1(1)       0       -1   0;      [t1;      [-x1(1);
%        0       x2(2)-x2(1)  -1   0;   *   t2;   =   -x2(1);
%   y1(2)-y1(1)       0        0  -1;       x0;       -y1(1);
%        0       y2(2)-y2(1)   0  -1]       y0]       -y2(1)]
%
% Let's call that A*T = B.  We can solve for T with T = A\B.
%
% Once we have our solution we just have to look at t1 and t2 to determine
% whether L1 and L2 intersect.  If 0 <= t1 < 1 and 0 <= t2 < 1 then the two
% line segments cross and we can include (x0,y0) in the output.
%
% In principle, we have to perform this computation on every pair of line
% segments in the input data.  This can be quite a large number of pairs so
% we will reduce it by doing a simple preliminary check to eliminate line
% segment pairs that could not possibly cross.  The check is to look at the
% smallest enclosing rectangles (with sides parallel to the axes) for each
% line segment pair and see if they overlap.  If they do then we have to
% compute t1 and t2 (via the A\B computation) to see if the line segments
% cross, but if they don't then the line segments cannot cross.  In a
% typical application, this technique will eliminate most of the potential
% line segment pairs.


% Input checks.
error(nargchk(2,5,nargin))

% Adjustments when fewer than five arguments are supplied.
switch nargin
	case 2
		robust = true;
		x2 = x1;
		y2 = y1;
		self_intersect = true;
	case 3
		robust = x2;
		x2 = x1;
		y2 = y1;
		self_intersect = true;
	case 4
		robust = true;
		self_intersect = false;
	case 5
		self_intersect = false;
end

% x1 and y1 must be vectors with same number of points (at least 2).
if sum(size(x1) > 1) ~= 1 || sum(size(y1) > 1) ~= 1 || ...
		length(x1) ~= length(y1)
	error('X1 and Y1 must be equal-length vectors of at least 2 points.')
end
% x2 and y2 must be vectors with same number of points (at least 2).
if sum(size(x2) > 1) ~= 1 || sum(size(y2) > 1) ~= 1 || ...
		length(x2) ~= length(y2)
	error('X2 and Y2 must be equal-length vectors of at least 2 points.')
end


% Force all inputs to be column vectors.
x1 = x1(:);
y1 = y1(:);
x2 = x2(:);
y2 = y2(:);

% Compute number of line segments in each curve and some differences we'll
% need later.
n1 = length(x1) - 1;
n2 = length(x2) - 1;
xy1 = [x1 y1];
xy2 = [x2 y2];
dxy1 = diff(xy1);
dxy2 = diff(xy2);

% Determine the combinations of i and j where the rectangle enclosing the
% i'th line segment of curve 1 overlaps with the rectangle enclosing the
% j'th line segment of curve 2.
[i,j] = find(repmat(min(x1(1:end-1),x1(2:end)),1,n2) <= ...
	repmat(max(x2(1:end-1),x2(2:end)).',n1,1) & ...
	repmat(max(x1(1:end-1),x1(2:end)),1,n2) >= ...
	repmat(min(x2(1:end-1),x2(2:end)).',n1,1) & ...
	repmat(min(y1(1:end-1),y1(2:end)),1,n2) <= ...
	repmat(max(y2(1:end-1),y2(2:end)).',n1,1) & ...
	repmat(max(y1(1:end-1),y1(2:end)),1,n2) >= ...
	repmat(min(y2(1:end-1),y2(2:end)).',n1,1));

% Force i and j to be column vectors, even when their length is zero, i.e.,
% we want them to be 0-by-1 instead of 0-by-0.
i = reshape(i,[],1);
j = reshape(j,[],1);

% Find segments pairs which have at least one vertex = NaN and remove them.
% This line is a fast way of finding such segment pairs.  We take
% advantage of the fact that NaNs propagate through calculations, in
% particular subtraction (in the calculation of dxy1 and dxy2, which we
% need anyway) and addition.
% At the same time we can remove redundant combinations of i and j in the
% case of finding intersections of a line with itself.
if self_intersect
	remove = isnan(sum(dxy1(i,:) + dxy2(j,:),2)) | j <= i + 1;
else
	remove = isnan(sum(dxy1(i,:) + dxy2(j,:),2));
end
i(remove) = [];
j(remove) = [];

% Initialize matrices.  We'll put the T's and B's in matrices and use them
% one column at a time.  AA is a 3-D extension of A where we'll use one
% plane at a time.
n = length(i);
T = zeros(4,n);
AA = zeros(4,4,n);
AA([1 2],3,:) = -1;
AA([3 4],4,:) = -1;
AA([1 3],1,:) = dxy1(i,:).';
AA([2 4],2,:) = dxy2(j,:).';
B = -[x1(i) x2(j) y1(i) y2(j)].';

% Loop through possibilities.  Trap singularity warning and then use
% lastwarn to see if that plane of AA is near singular.  Process any such
% segment pairs to determine if they are colinear (overlap) or merely
% parallel.  That test consists of checking to see if one of the endpoints
% of the curve 2 segment lies on the curve 1 segment.  This is done by
% checking the cross product
%
%   (x1(2),y1(2)) - (x1(1),y1(1)) x (x2(2),y2(2)) - (x1(1),y1(1)).
%
% If this is close to zero then the segments overlap.

% If the robust option is false then we assume no two segment pairs are
% parallel and just go ahead and do the computation.  If A is ever singular
% a warning will appear.  This is faster and obviously you should use it
% only when you know you will never have overlapping or parallel segment
% pairs.

if robust
	overlap = false(n,1);
	warning_state = warning('off','MATLAB:singularMatrix');
	% Use try-catch to guarantee original warning state is restored.
	try
		lastwarn('')
		for k = 1:n
			T(:,k) = AA(:,:,k)\B(:,k);
			[unused,last_warn] = lastwarn;
			lastwarn('')
			if strcmp(last_warn,'MATLAB:singularMatrix')
				% Force in_range(k) to be false.
				T(1,k) = NaN;
				% Determine if these segments overlap or are just parallel.
				overlap(k) = rcond([dxy1(i(k),:);xy2(j(k),:) - xy1(i(k),:)]) < eps;
			end
		end
		warning(warning_state)
	catch err
		warning(warning_state)
		rethrow(err)
	end
	% Find where t1 and t2 are between 0 and 1 and return the corresponding
	% x0 and y0 values.
	in_range = (T(1,:) >= 0 & T(2,:) >= 0 & T(1,:) <= 1 & T(2,:) <= 1).';
	% For overlapping segment pairs the algorithm will return an
	% intersection point that is at the center of the overlapping region.
	if any(overlap)
		ia = i(overlap);
		ja = j(overlap);
		% set x0 and y0 to middle of overlapping region.
		T(3,overlap) = (max(min(x1(ia),x1(ia+1)),min(x2(ja),x2(ja+1))) + ...
			min(max(x1(ia),x1(ia+1)),max(x2(ja),x2(ja+1)))).'/2;
		T(4,overlap) = (max(min(y1(ia),y1(ia+1)),min(y2(ja),y2(ja+1))) + ...
			min(max(y1(ia),y1(ia+1)),max(y2(ja),y2(ja+1)))).'/2;
		selected = in_range | overlap;
	else
		selected = in_range;
	end
	xy0 = T(3:4,selected).';
	
	% Remove duplicate intersection points.
	[xy0,index] = unique(xy0,'rows');
	x0 = xy0(:,1);
	y0 = xy0(:,2);
	
	% Compute how far along each line segment the intersections are.
	if nargout > 2
		sel_index = find(selected);
		sel = sel_index(index);
		iout = i(sel) + T(1,sel).';
		jout = j(sel) + T(2,sel).';
	end
else % non-robust option
	for k = 1:n
		[L,U] = lu(AA(:,:,k));
		T(:,k) = U\(L\B(:,k));
	end
	
	% Find where t1 and t2 are between 0 and 1 and return the corresponding
	% x0 and y0 values.
	in_range = (T(1,:) >= 0 & T(2,:) >= 0 & T(1,:) < 1 & T(2,:) < 1).';
	x0 = T(3,in_range).';
	y0 = T(4,in_range).';
	
	% Compute how far along each line segment the intersections are.
	if nargout > 2
		iout = i(in_range) + T(1,in_range).';
		jout = j(in_range) + T(2,in_range).';
	end
end
end

% Plot the results (useful for debugging).
% plot(x1,y1,x2,y2,x0,y0,'ok');

function [Y] = adjustDimensions(X,p)
%Adjust the dimensions of X to pxp
m = size(X,1);
Y = zeros(p,p);
if(p > m)
    newRows = zeros(p-m,m);
    newCols = zeros(p,p-m);
    temp = [X;newRows];
    Y = [temp newCols];
else
    if(p == m)
        Y = X;
    else    
        Y(:,:) = X(1:p,1:p);
    end
end
end


function [R] = measurementNoiseNew(xseg,fs)
% method for calculating measurement noise variance based on PSD

numFrame = size(xseg,1);
noise_covariance = zeros(1,numFrame);
spec_flat = zeros(1,numFrame);

for k = 1:numFrame
    [c, lag] = xcorr(xseg(k,:),'coeff');
    %calculating power spectral density from ACF
    psd = (fftshift(abs(fft(c))));
    psd = psd(round(length(psd)/2):end);
    freq = (fs * (0:length(c)/2))/length(c);
    %keep positive lags only since ACF is symmetrical
    c = c(find(lag == 0):length(c));
    lag = lag(find(lag == 0):length(lag));
    %keep frequencies from 100Hz to 2kHz
    freq2 = find(freq>= 100 & freq<=2000);
    psd2 = psd(freq2);
    spec_flat(k) = geomean(psd2)/mean(psd2);
end

normal_flat = spec_flat/max(spec_flat);%normalize spectral flatness
thr = 0.707; %threshold
for k = 1:numFrame
    if normal_flat(k) >= thr
        noise_covariance(k) = var(xseg(k,:));
    end
end
R= max(noise_covariance);
end

function order = findOrder(n, ~, ~)
%estimates order of each frame of noisy signal
total = size(n,1);
order = zeros(total,1);
T=100;

for i = 1:total
    [coef,noisevar,reflection_coefs] = aryule(n(i,:),T); 
    pacf = -reflection_coefs;
    cpacf = cumsum(abs(pacf));
%estimated order = lag at which CPACF is 70% of range of CPACF 
    dist = abs(cpacf - 0.7*(range(cpacf)));
    order(i) = find(dist == min(dist),1,'first');
    if i==4 || i==total-1 
        if i == 4
            figure(5);
            heading = 'PACF plot for Voiced Frame';
        else
            figure(6);
            heading = 'PACF plot for Silent Frame';
        end
        title(heading);
        subplot(2,1,1);
        stem(pacf,'filled','MarkerSize',4); 
        xlabel('Lag');
        ylabel('Partial-Autocorrelation Coefficients (PACF)'); 
        xlim([1 T]);
        u_confidence = 1.96/sqrt(size(n,2));
        l_confidence = -u_confidence;
        hold on;
        plot([1 T],[1 1]'*[l_confidence u_confidence],'r');
        hold off;
        subplot(2,1,2);
        text = ['Estimated order = ',num2str(order(i))]; 
        stem(cpacf,'filled','MarkerSize',4); 
        xlabel('Lag');
        ylabel('Cumulative PACF');
        title(text);
        grid on;
        hold on;
        plot(0.7*range(cpacf)*ones(1,T),'r');
        hold off;
        xlabel('Lags');
        ylabel('Cumulative PACF');
    end
end
end

function [cleanSpeech] = cmi_tuned_kalman_speech(x,fs)

%Inputs:
%x - noisy speech sample
%fs - sampling rate
%Output:
%cleanSpeech - enhanced speech

p=15;%order of lpc
y=x';

%dividing into overlapping 80ms frames
start=1;
l=0.08*fs;
overlap=0.01*fs;
totseg=ceil(length(y)/(l-overlap));
segment=zeros(totseg,l);

for i=1:totseg-1
    segment(i,1:l)=y(1,start:start+l-1);
    start=(l-overlap)*i+1;
end
segment(totseg,1:length(y)-start+1)=y(start:length(y));


H=[zeros(1,p-1),1];
G=H';
cleanspeech=zeros(totseg,l);
cleanSpeech=zeros(1,length(y));


[R, silent_inds] = measurementNoiseNew(segment, fs);       
J1=zeros(1,10);
J2=zeros(1,10);
nq=zeros(1,10);
u=1;
X=y(1:p)';
P=zeros(l,p,p);
P(1,:,:)=R*eye(p);
Q=0;
t1=zeros(p,p);

K_Q2=zeros(p,l,totseg);
K_Q3=zeros(p,l,totseg);
Kavg_Q2=zeros(1,totseg);
Kavg_Q3=zeros(1,totseg);
J1_Q3=zeros(1,totseg);
J1_Q2=zeros(1,totseg);
Xhat_Q2=zeros(p,l,totseg);
Xhat_Q3=zeros(p,l,totseg);


for i=1:totseg
    
    %first iteration of Kalman filter
    [A,Q1]=lpc(segment(i,:),p);
    temp=eye(p);
    PHI=[temp(2:p,:);-fliplr(A(2:end))];
    
    %tuning the filter by calculating optimum value of process noise
    %variance
   q=1;
   if i~=1

     for n=-5:4
       Q0=(10^n)*Q1;
       t1(:,:)=P(1,:,:);
       Ak=H*(PHI*t1*PHI')*H';
       Bk=H*Q0*H';
       J1(q)=R/(Ak+Bk+R);
       J2(q)=Bk/(Ak+Bk);
       nq(q)=log10(Bk);
       q=q+1;
     end
     
     
     [nq_nom,J1_Q2(i)]=intersections(nq,J1,nq,J2);
     
     %sensitivity metrics gets higher preference and we select Q<Qcomp
     if numel(nq_nom)~=0
            Q=10^(nq_nom-0.7);
     else
       Q=Q1;
     end
        
   else
     Q=Q1;
   end
   u=u+1;
   
   for j=1:length(segment(i,:))
        X_=PHI*X;
        t1(:,:)=P(j,:,:);
        P_=(PHI*t1*PHI')+(G*Q*G');
        K_Q2(:,j,i)=(P_*H')*(inv(H*P_*H'+R));
        t1=(temp-K_Q2(:,j,i)*H)*P_;
        P(j+1,:,:)=t1(:,:);
        e=segment(i,j)-(H*X_);
        X=X_+K_Q2(:,j,i)*e;
        cleanspeech(i,j)=X(end);
        Xhat_Q2(:,j,i)=X_;
        
   end
    Kavg_Q2(i)=mean(K_Q2(p,:,i));
    P(1,:,:)=P(j-1,:,:);
       
    
end
u=1;

 for i=1:totseg
    
    %first iteration of Kalman filter
    [A,Q1]=lpc(segment(i,:),p);
    temp=eye(p);
    PHI=[temp(2:p,:);-fliplr(A(2:end))];
    
    %tuning the filter by calculating optimum value of process noise
    %variance
   q=1;
   if i~=1

     for n=-5:4
       Q0=(10^n)*Q1;
       t1(:,:)=P(1,:,:);
       Ak=H*(PHI*t1*PHI')*H';
       Bk=H*Q0*H';
       J1(q)=R/(Ak+Bk+R);
       J2(q)=Bk/(Ak+Bk);
       nq(q)=log10(Bk);
       q=q+1;
     end
     
     
     [nq_nom,J1_Q3(i)]=intersections(nq,J1,nq,J2);
     
     %sensitivity metrics gets higher preference and we select Q<Qcomp
     if numel(nq_nom)~=0
            Q=10^(nq_nom);
     else
       Q=Q1;
     end
        
   else
     Q=Q1;
   end
   u=u+1;
   
   for j=1:length(segment(i,:))
        X_=PHI*X;
        t1(:,:)=P(j,:,:);
        P_=(PHI*t1*PHI')+(G*Q*G');
        K_Q3(:,j,i)=(P_*H')*(inv(H*P_*H'+R));
        t1=(temp-K_Q3(:,j,i)*H)*P_;
        P(j+1,:,:)=t1(:,:);
        e=segment(i,j)-(H*X_);
        X=X_+K_Q3(:,j,i)*e;
        cleanspeech(i,j)=X(end);
        Xhat_Q3(:,j,i)=X_;
        
   end
    Kavg_Q3(i)=mean(K_Q3(p,:,i));
    P(1,:,:)=P(j-1,:,:);
        
 end

K=zeros(1,totseg);
I=ones(p,1);

for i=1:totseg
         
      if(silent_inds(i) == 1)
       %silent region
          for j=1:length(segment(i,:))
              X=K_Q2(:,j,i)*segment(i,j)+(I-K_Q2(:,j,i))*(H*Xhat_Q2(:,j,i));
              cleanspeech(i,j)=X(end);
          end
          K(i)=Kavg_Q2(i);
      else
      %voiced region
          for j=1:length(segment(i,:))
              X=K_Q3(:,j,i)*segment(i,j)+(I-K_Q3(:,j,i))*(H*Xhat_Q3(:,j,i));
              cleanspeech(i,j)=X(end);
          end
           K(i)=Kavg_Q3(i);
      end
     
     %second iteration of kalman filter with lpcs calculated from cleaned speech      
     [A,Q]=lpc(cleanspeech(i,:),p);
     temp=eye(p);
     PHI=[temp(2:p,:);-fliplr(A(2:end))];
     if i==1
        X=cleanspeech(i,1:p)';
        P0=temp*R;
     end
    
    for j=1:length(segment(i,:))
        
        X_=PHI*X;
        P_=(PHI*P0*PHI')+(G*Q*G');
        K0=(P_*H')*(inv(H*P_*H'+R));
        P0=(eye(p)-K0*H)*P_;
        e=segment(i,j)-(H*X_);
        X=X_+K0*e;
        cleanspeech(i,j)=X(end);
    end
       
end



%overlap add
cleanSpeech(1:l)=cleanspeech(1,1:l);
start=l+1;
for i=2:totseg-1
    cleanSpeech(start:start+(l-overlap))=cleanspeech(i,overlap:end);
    start=start+l-overlap-1;
end
cleanSpeech(start:length(y))=cleanspeech(totseg,1:(length(y)-start)+1);
cleanSpeech=cleanSpeech(1:length(y));
   

end
function [cleanSpeech] = kalman_speech_varQ(x,fs)

%Tuned Kalman filter for speech enhancement with different values of
%process noise covariance

%Inputs:
%x - noisy speech sample
%fs - sampling rate
%Output:
%cleanSpeech - enhanced speech

prompt='What is the value of Q? 0 for Qnom(from lpc),1 for Q1,2 for Q2,3 for Qc,4 for Q3,5 for Q4      ';
in=input(prompt);

while in<0 || in>5
    disp('Wrong input,input must be between 0 and 5');
    in=input(prompt);
    if in>=0 && in<=5
        break;
    end
end
    
Q_pos=[-3,-0.7,0,1,3];

p=15;%order of lpc
y=x';

%dividing into overlapping 80ms frames
start=1;
l=0.08*fs;
overlap=0.01*fs;
totseg=ceil(length(y)/(l-overlap));
segment=zeros(totseg,l);

for i=1:totseg-1
    segment(i,1:l)=y(1,start:start+l-1);
    start=(l-overlap)*i+1;
end
segment(totseg,1:length(y)-start+1)=y(start:length(y));

H=[zeros(1,p-1),1];
G=H';
cleanspeech=zeros(totseg,l);
cleanSpeech=zeros(1,length(y));


R = measurementNoiseNew(segment,fs);
J1=zeros(1,10);
J2=zeros(1,10);
nq=zeros(1,10);
u=1;

X=y(1:p)';
P=zeros(l,p,p);
P(1,:,:)=R*eye(p);
t1=zeros(p,p);
Q_arr=zeros(1,totseg);

for i=1:totseg
    
    %first iteration of Kalman filter
    [A,Q1]=lpc(segment(i,:),p);
    temp=eye(p);
    PHI=[temp(2:p,:);-fliplr(A(2:end))];
    
    %tuning the filter by calculating optimum value of process noise
    %variance
   q=1;
   if i~=1 && in~=0

     for n=-5:4
       Q0=(10^n)*Q1;
       t1(:,:)=P(1,:,:);
       Ak=H*(PHI*t1*PHI')*H';
       Bk=H*Q0*H';
       J1(q)=R/(Ak+Bk+R);
       J2(q)=Bk/(Ak+Bk);
       nq(q)=log10(Bk);
       q=q+1;
     end
     
     
     [nq_nom,J]=intersections(nq,J1,nq,J2);
     
     %sensitivity metrics gets higher preference and we select Q<Qcomp
     if numel(nq_nom)~=0
            Q=10^(nq_nom+Q_pos(in));
     else
       Q=Q1;
     end
     
   else
       Q=Q1;
   end
   Q_arr(u)=Q;
   u=u+1;
   
   for j=1:length(segment(i,:))
        X_=PHI*X;
        t1(:,:)=P(j,:,:);
        P_=(PHI*t1*PHI')+(G*Q*G');
        K=(P_*H')*(inv(H*P_*H'+R));
        t1=(eye(p)-K*H)*P_;
        P(j+1,:,:)=t1(:,:);
        e=segment(i,j)-(H*X_);
        X=X_+K*e;
        cleanspeech(i,j)=X(end);
        
   end
    P(1,:,:)=P(j-1,:,:);
       
    %second iteration of Kalman filter with lpc calculated from
    %cleaned speech
    
    [A,Q]=lpc(cleanspeech(i,:),p);
    PHI=[temp(2:p,:);-fliplr(A(2:end))];
     if i==1
        X=cleanspeech(i,1:p)';
        P0=temp*R;
     end
    
    for j=1:length(segment(i,:))
        
        X_=PHI*X;
        P_=(PHI*P0*PHI')+(G*Q*G');
        K=(P_*H')*(inv(H*P_*H'+R));
        P0=(eye(p)-K*H)*P_;
        e=segment(i,j)-(H*X_);
        X=X_+K*e;
        cleanspeech(i,j)=X(end);
                           
    end

end

%overlap add
cleanSpeech(1:l)=cleanspeech(1,1:l);
start=l+1;
for i=2:totseg-1
    cleanSpeech(start:start+(l-overlap))=cleanspeech(i,overlap:end);
    start=start+l-overlap-1;
end
cleanSpeech(start:length(y))=cleanspeech(totseg,1:(length(y)-start)+1);
cleanSpeech=cleanSpeech(1:length(y));

end
