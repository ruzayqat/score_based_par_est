%In this code, we try to estimate the parameters that the observations were
%generated from using the first algorithm (Alg 3.1) proposed in our paper:
%"Beskos, A., Crisan, D., Jasra, A., Kantas, N. & Ruzayqat, H. (2020). 
%Score-Based Parameter Estimation for a Class of Continuous-Time State
%Space Models. arXiv:2008.07803v1"
%We apply this on the first model in that paper:
%Let dx = dy = 1, d_theta = 2 and consider the following linear SDE:
%       dXt = theta_1 * Xt * dt + sigma * dWt;
%       dYt = thata_2 *(kappa -  Xt) * dt + dBt:
% where {Wt}_t and {Bt} re two independent Brownian motions. kappa and
% sigma are fixed and theta_1, theta_2 are the parameters to be estimated.

clc
close all
clear
format long;


T = 20000; %number of iterations (# of time intervals)
Lmax = 13; %the level at which the data will be generated
N = 2500; %Number of particles to use
resamp_coef = 0.8; %for resampling

stepsize_power = -0.8; %for stochastic gradient method

theta0 = [-0.1; -1.5]; %initial guess 

theta_true = [-0.7; -0.5]; %true parameters to be estimated

kappa = 2; %fixed 
sig = 0.3; %fixed and it is a constant diffusion

X_star = 0.2; %X(0)





%% Generate the observations data
y_Lmax = zeros(2^Lmax * T + 1, 1); %observations array
delta = 2^(-Lmax);                 
y_Lmax(1) = 0; %Y is a Brownian motion under P^{\bar}
X = X_star;
for n = 2: 2^Lmax * T + 1
    dV = sqrt(delta) * randn;
    X = X + X * theta_true(1) * delta + sig * dV; 
    dB = sqrt(delta) * randn;
    y_Lmax(n) = y_Lmax(n-1) + theta_true(2) * (kappa - X) * delta + dB;
end


%% Set level L to run the algorithm at. Cut the data points on this level
L = 10;
s = 2^(Lmax-L);
y = y_Lmax(1:s:end); %its size = T*2^L + 1



%% Start the algorithm
%initialize the matrices that will have the score and parameter estimates
grad_log = zeros(T,2); 
theta = zeros(T,2);

%initialize the N particles positions
X0 = X_star * ones(N,1);
%Brownian motion at time = 0;
W0 = zeros(N,1); 
%Later on W0 will be the Brownian motion at the start point of each time
%interval

%initialize the matrix F
F = zeros(N,2);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%at n = 0 (First time interval)
i1 = 1;
i2 = 2^L+1;
y0 = y(i1:i2);%data at points between x_Delta_l & x_1 at level l = L

%Call the function "sample_calc_G" to return the Brownian motion, the
%unnormalized weight G and the paths of all particles on the first interval
%The unnormalized weight G is $G_{0,\theta}^l$ in Eq. 3.8 in the paper.

[W, G, ~, X] = sample_calc_G(L, N, X0, W0, y0, sig, theta0, kappa);

%Call the function "calc_Lambda_1st_interval" to calculate Lambda on the
%first interval. Note that calculating Lambda on the first interval is a
%little different from the other intervals because all particles share the
%same initial position X_star. 

Lambda = calc_Lambda_1st_interval(L, N, X0, X, W0, W,y0,sig,theta0,kappa);

%normalize the weight
weight = G/sum(G); 
    
%estimate the gradient of the log-likelihood on the first interval    
grad_log(1,:) = sum(weight .* Lambda);    

%Set theta at iteration 1 to be theta0
theta(1,:) = theta0; 

%****************
%Set things for the next iteration and resample if neccessary
X0 = X(:,end); 
W0 = W(:,end); 

ESS = (sum(weight.^2))^(-1); %effective sample size
if ESS <= resamp_coef * N
    In = randsample(N,N,true,weight); %return resampled indices
    X0 = X0(In); %resample the particles
    W0 = W0(In); %re-order the Brownian motions accordingly
    Lambda = Lambda(In,:); %re-order Lambda accordingly   
    %set the weights to be 1/N
    weight = ones(N,1)/N;
end
%Save Lambda in F_old to be used in the next iteration
F_old = Lambda;

%print at iteration 1
fprintf('T = 1, theta =[ %.5f, %.5f\n', theta(1,:))
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Iterate 
for n = 1:T-1
    i1 = n * 2^L + 1;
    i2 = (n+1) * 2^L + 1;
    y0 = y(i1:i2);%data at points between x_{n+Delta_l) & x_{n} at level L

    %Call the function "sample_calc_G" to return the Brownian motion, the
    %unnormalized weight G, little g (See Eq. 3.9 in the paper) and the
    %paths of all particles on the first interval.
    %The unnormalized weight G is $G_{0,\theta}^l$ in Eq. 3.8 in the paper.
    [W, G, g, X] = sample_calc_G(L, N, X0, W0, y0, sig, theta(n,:), kappa);

    %Update the weight
    weight = weight .* G;
    %normalize the weight
    weight = weight/sum(weight);

    X_Delta = X(:,1);
    %the mean and the standard deviation needed for the Euler transition
    %density evaluated at X(:,1):
    mu = X0 + theta(n,1) * X0 * 2^(-L);
    std = sig * 2^(-L/2); %= sqrt(variance) = sqrt(sig^2 *2^(-L))
    %calculate F. We do this in a vectorized way.
    for i = 1:N
        Wi = W(i,:); %Brownian motion for particle i from
                     %t = n+Delta_l to t = n+1
        Xi = X(i,:); % path of particle i from t = n+Delta_l to t = n+1
        Lambda = calc_Lambda(L,N,X0,W0,Xi,Wi,y0,sig,theta(n,:),kappa);
        %Due to cancelations, we only need the following:
        m = normpdf(X_Delta(i),mu,std);
        g_timesP = g .* m;
        ssum = sum(g_timesP);
        F(i,:) = sum(g_timesP .* (Lambda + F_old)) / ssum;
    end

    %estimate the gradient of the log-likelihood
    grad_log(n+1,:) = sum(weight .* F);


    diff_grad = grad_log(n+1,:) - grad_log(n,:);
    
    %update theta
    step = (n+1)^stepsize_power;
    theta(n+1,:) = theta(n,:) + step * diff_grad;
    
    %*********************
    %Set things for the next iteration and resample if neccessary
    X0 = X(:,end);
    W0 = W(:,end);
    ESS = (sum(weight.^2))^(-1); %effective sample size
    if ESS <= resamp_coef * N
        In = randsample(N,N,true,weight); %return resampled indices
        X0 = X0(In);  %resample the particles
        W0 = W0(In); %re-order the Brownian motions accordingly
        F = F(In,:); %re-order Lambda accordingly
        %set the weights to be 1/N
        weight = ones(N,1)/N;
    end
    %Save F for the next iteration
    F_old = F;
    
    %print
    fprintf('T = %d, theta = [%.5f,  %.5f]\n', n+1, theta(n+1,:))

end %n=1:T-1


%% plotting


%set the background of the figure to be white
set(0,'defaultfigurecolor',[1 1 1])

set(0, 'defaultLegendInterpreter','latex');
set(0, 'defaultTextInterpreter','latex');

%plot the values of theta over time
Time2 = 1:T;

figure

txt1 = '$\theta_1$';
plot(Time2,theta(:,1),'r-','Linewidth',2,'DisplayName',txt1);
hold on

txt2 = 'True $\theta_1$ ';
yline(theta_true(1),'k--','DisplayName',txt2,'Linewidth',1.8) ; 
hold on

txt3 = '$\theta_2$';
plot(Time2,theta(:,2),'b-','Linewidth',2,'DisplayName',txt3);
hold on

txt4 = 'True $\theta_2$ ';
yline(theta_true(2),'g--','DisplayName',txt4,'Linewidth',1.8) ;

hold off

grid on
legend show
legend('Location','northwest')
set(legend, 'FontSize', 16)
xlabel('Time')
ylabel('$\theta$')

set(gcf, 'Position', [500 100 1200 700]);
set(gca,'FontSize',18);



%% Functions

function Lambda = calc_Lambda(L,N,X0,W0,Xi,Wi,y,sig,theta,kappa)
    p = 2^L;
    Delta = 1/p;
    
    Lambda = zeros(N,2);
    kappa_minus_x = kappa -X0;
    
    Lambda(:,1) = sig^(-1) * X0 .* (Wi(1) - W0); %note this is a vector
    Lambda(:,2) = kappa_minus_x * (y(2)-y(1)) - theta(2) * Delta ...
                    * kappa_minus_x.^2;
    temp1 = 0;
    temp2 = 0;
    for j = 1:p-1
        temp1 = temp1 + sig^(-1) * Xi(j) * (Wi(j+1) - Wi(j));
        kappa_minus_x = kappa - Xi(j);
        temp2 = temp2 + kappa_minus_x * (y(j+2) - y(j+1)) - ...
                theta(2) * Delta * kappa_minus_x.^2;
    end
    
    Lambda(:,1) = Lambda(:,1) + temp1 * ones(N,1);
    Lambda(:,2) = Lambda(:,2) + temp2 * ones(N,1);
end


function Lambda = calc_Lambda_1st_interval(L,N,X0,X,W0,W,y,sig,theta,kappa)
    p = 2^L;
    Delta = 1/p;
    
    Lambda = zeros(N,2);
    kappa_minus_x = kappa -X0;
    
    Lambda(:,1) = sig^(-1) * X0 .* (W(:,1)- W0);
    Lambda(:,2) = kappa_minus_x * (y(2)-y(1)) - theta(2) * Delta ...
                    * kappa_minus_x.^2;
    for i = 1:p-1
        Lambda(:,1) = Lambda(:,1) + sig^(-1) * X(:,i) .* (W(:,i+1)-W(:,i));
        kappa_minus_x = kappa -X(:,i);
        Lambda(:,2) = Lambda(:,2) + kappa_minus_x * (y(i+2) - y(i+1)) - ...
                    theta(2) * Delta * kappa_minus_x.^2;
    end
end



function [W, G, g, X] = sample_calc_G(L, N, X0, W0, y0, sig, theta, kappa)
% This won't work if l = 0, p can't be 1. 
% In MLPF, need to take this into consideration.


p = 2^L;
Delta = 1/p;

Gt = zeros(N,p);
W = zeros(N,p);
X = zeros(N,p);

kappa_minus_X = kappa -X0;
Gt(:,1) = theta(2) * kappa_minus_X * (y0(2)-y0(1)) - 0.5 * Delta * ...
            theta(2)^2 * kappa_minus_X.^2;

epsilon = sqrt(Delta) * randn(N,1);
W(:,1) = W0 + epsilon;
X(:,1) = X0 + X0 * theta(1) * Delta + sig * epsilon; 

kappa_minus_X = kappa -X(:,1);
Gt(:,2) = theta(2) * kappa_minus_X * (y0(3)-y0(2)) - 0.5 * Delta * ...
             theta(2)^2 * kappa_minus_X.^2;

for i = 1:p-2
    epsilon = sqrt(Delta) * randn(N,1);
    W(:,i+1) = W(:,i) + epsilon;
    X(:,i+1) = X(:,i) + X(:,i) * theta(1) * Delta + sig * epsilon; 
    
    kappa_minus_X = kappa -X(:,i+1);
    Gt(:,i+2) = theta(2) * kappa_minus_X * (y0(i+3)-y0(i+2)) - 0.5 * ...
            Delta * theta(2)^2 * kappa_minus_X.^2;
end
epsilon = sqrt(Delta) * randn(N,1);
W(:,p) = W(:,p-1) + epsilon;
X(:,p) = X(:,p-1) + X(:,p-1) * theta(1) * Delta + sig * epsilon;

g = exp(Gt(:,1));
G = exp(sum(Gt,2));
end
