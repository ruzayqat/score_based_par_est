%In this code, we try to estimate the parameters that the observations were
%generated from using the multilevel version of the second algorithm 
% (Alg 4.3) proposed in our paper:
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
l_min = 7; %This is l_star - 1  in the paper
rho = 0.14; % We need it for the number of particles on each level Nl

resamp_coef = 0.8; %for resampling

stepsize_power = -0.8; %for stochastic gradient method

theta0 = [-0.1; -1.5]; %initial guess 

theta_true = [-0.7; -0.5]; %true parameters to be estimated

kappa = 2; %fixed 
sig = 0.3; %fixed and it is a constant diffusion

X_star = 0.2; %X(0)


%% Generate the observations data

y_Lmax = zeros(2^Lmax * T + 1, 1);
delta = 2^(-Lmax);                 
y_Lmax(1) = 0; %Y is a Brownian motion under P^{\bar}
X = X_star;

for n = 2: 2^Lmax * T + 1
    dV = sqrt(delta) * randn;
    X = X + X * theta_true(1) * delta + sig * dV; 
    dB = sqrt(delta) * randn;
    y_Lmax(n) = y_Lmax(n-1) + theta_true(2) * (kappa - X) * delta + dB;

end

%USe the same data that was used in Alg 3.1
% fileID = fopen('data_Ex1_Xstar_0_2_theta-0_7-0_5_T20000_Lmax10.txt','r');
% formatSpec = '%f';
% rows = 2^Lmax * T + 1;
% sizeA = [rows 1];
% y_Lmax = fscanf(fileID,formatSpec, sizeA);

  
%% Set level L to run the algorithm at. 
% Cut the data points for each level l = l_min up to l = L

L = 10;

y_l = cell(L,1);
%y_l{l} = the data on level l
for l = l_min : L
  s = 2^(Lmax-l);
  y_l{l} = y_Lmax(1:s:end); %its size = T*2^l + 1
end


%% Start the algorithm
%initialize

Delta_l = zeros(L,1); 
for l = 1:L
  Delta_l(l) = 2^(-l);
end

%Generate the number of particles on each level l
Nl =  floor( 2^(L) * (L-l_min+1) * Delta_l.^(0.5+rho));

%Note we only going to need Nl(l_min:L)

%initialize Aln, where Aln(l_min) = estimate of the score function on level
%l_min, and Aln(l) (for l>l_min) is the difference between the estimate of
%the score function on the finer level l and the coearser level l-1
Aln = zeros(L,2);

%initialize the matrices that will have the score and parameter estimates
grad_log_ml = zeros(T,2);
theta = zeros(T,2);


X0_f = cell(L,1);
%X0_f{l} = the start point of the paths of the particles at the beginning
%of each time interval on level l
for l = l_min:L
    X0_f{l} = X_star * ones(Nl(l),1); 
end 

weight_f = cell(L,1);

F_old_f = cell(L,1);
F_old_c = cell(L,1);

X0_c = X0_f;
weight_c = weight_f;


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%at n = 0 (First time interval)
for l = l_min : L
  if l == l_min
    i1 = 1;
    i2 = 2^l+1;
    y = y_l{l}(i1:i2);%data at points between x_Delta_l & x_1 at level l 

    %Since sigma is constant, then \hat{p} is the same as \tilde{p}, which
    %is N(X0, sig^2);
    %sample X1 from the distribution N(X0, sig^2) for all particles
    %in a column vector. Note in MATLAB, in 1d, normrnd takes the standard
    %deviations and not the variance.
    X1 = normrnd(X0_f{l},sig);

    %Generate the Brownian increments by sampling them from a normal
    %distribution N(0, Delta_l). Note in MATLAB, randn(rows, colns) returns
    %a matrix of size rows x colns of random numbers sampled from the
    %standard normal distribution.
    Z = sqrt(Delta_l(l)) * randn(Nl(l),2^l-1);

    %call the function "solve_Bridge_sde" to return the particles paths X 
    %on the current time interval (Eq. 4.11) as a matrix of size
    % Nl(l) x 2^l, where X(:,2^l) = X1. And it will also return the
    %unnormalized weight \tilde{G}_{0,\theta}^l$ in Eq. 4.14 for all 
    %particles
    [X, G] = solve_Bridge_sde(l, Nl(l), X0_f{l}, X1, Z, y,...
                    sig, theta0, kappa);

    %There is no N^2 recursion in the first interval. 
    Lambda = calc_Lambda(l, Nl(l), X0_f{l}, X, y, sig, theta0, kappa);

    %calculate the normalized weight
    weight_f{l} = G/sum(G);
    
    %estimate of the gradient on level l_min
    Aln(l,:) = sum(weight_f{l} .* Lambda);
    
    %****************
    %Set things for the next iteration on the same level and 
    %resample if neccessary
    X0_f{l} = X1; 
    ESS = (sum(weight_f{l}.^2))^(-1); %Effective sample size
    if ESS <= resamp_coef * Nl(l)
        %return resampled indices
        In = randsample(Nl(l), Nl(l), true, weight_f{l}); 
        X0_f{l} = X0_f{l}(In); %resample the particles
        Lambda = Lambda(In,:); %re-order Lambda accordingly
        %set the weights to be 1/Nl(l)
        weight_f{l} = ones(Nl(l),1)/Nl(l);
    end
    %Save Lambda in F_old_f{l} to be used in the next iteration on the same
    %level
    F_old_f{l} = Lambda;

  else %l >= l_min+1 (still in the first interval)
    i1 = 1;
    i2 = 2^l+1;
    j1= 1;
    j2 = 2^(l-1)+1;

    %data at points between x_Delta_l & x_1 at level l 
    y_f = y_l{l}(i1:i2); 
    %data at points between x_Delta_{l-1} & x_1 at level l-1
    y_c = y_l{l-1}(j1:j2); 

    %sample X1 from the normal distribution N(X0, sig^2) for all particles
    %in a column vector
    X1_f = normrnd(X0_f{l},sig); 
    X1_c = X1_f; %on the first interval we set X1_c = X1_f

    %Generate the Brownian increments on the finer level
    Z_f = sqrt(Delta_l(l)) * randn(Nl(l),2^l-1);
    len = 2^(l-1)-1;
    %Couple the finer and the coarser Brownian increments
    Z_c = zeros(Nl(l),len);
    %Note: last increment in Z_f is not needed in Z_c
    for II = 1 : len
        for m =1:2
            k=2*(II-1)+m;
            Z_c(:,II) = Z_c(:,II) + Z_f(:,k);
        end    
    end


    %call the function "solve_Bridge_sde" to return the particles paths X_f 
    %on the current time interval (Eq. 4.11) on level l as a matrix of 
    % size Nl(l) x 2^l, where
    %X_f(:,2^l) = X1_f. And it will also return the unnormalized weight
    %$\tilde{G}_{0,\theta}^{l}$ in Eq. 4.17 for all particles
    [X_f, G_f] = solve_Bridge_sde(l, Nl(l), X0_f{l}, X1_f, Z_f, y_f,...
                    sig, theta0, kappa);

    %normalize the weight
    weight_f{l} = G_f/sum(G_f);
    
    %There is no N^2 recursion in the first interval
    Lambda_f = calc_Lambda(l, Nl(l), X0_f{l}, X_f,y_f, sig, theta0, kappa);

    %call the function "solve_Bridge_sde" to return the particles paths X_c 
    %on the current time interval (Eq. 4.11) on level l-1 as a matrix of 
    % size Nl(l) x 2^{l-1}, where X_c(:,2^{l-1}) = X1_c. And it will also
    %return the unnormalized weight $\tilde{G}_{0,\theta}^{l-1}$ in 
    %Eq. 4.17 for all particles
    [X_c, G_c] = solve_Bridge_sde(l-1, Nl(l), X0_c{l}, X1_c, Z_c, y_c,...
                    sig, theta0, kappa);
    %normalize the weights
    weight_c{l} = G_c/sum(G_c);
    %There is no N^2 recursion in the first interval
    Lambda_c = calc_Lambda(l-1,Nl(l), X0_c{l}, X_c, y_c,sig,theta0, kappa);
    
    %Update Aln
    Aln(l,:) = sum(weight_f{l} .* Lambda_f) - sum(weight_c{l} .* Lambda_c);

    %****************
    %Set things for the next iteration on the same level and 
    %resample if neccessary
    X0_f{l} = X1_f;
    X0_c{l} = X1_c; 

    % We take the effective sample size as following:
    ESS = min( (sum(weight_f{l}.^2))^(-1), (sum(weight_c{l}.^2))^(-1) );
    if ESS <= resamp_coef * Nl(l)
        %Call the function "coupled_resampling" to return a coupled indices
        %using maximal coupling
        [In_f, In_c] = coupled_resampling(Nl(l), weight_f{l}, weight_c{l});
        X0_f{l} = X0_f{l}(In_f); %resample particles on the finer level
        X0_c{l} = X0_c{l}(In_c);%resample particles on the coarser level

        Lambda_f = Lambda_f(In_f,:); %reorder Lambda_f accordingly
        Lambda_c = Lambda_c(In_c,:); %reorder Lambda_c accordingly

        %normalize the weights
        weight_f{l} = ones(Nl(l),1)/Nl(l);
        weight_c{l} = weight_f{l};
    end

    %Save Lambda_f and Lambda_c in F_old_f{l} and F_old_c{l} to be used 
    %in the next iteration on the same level
    F_old_f{l} = Lambda_f;
    F_old_c{l} = Lambda_c;

  end %if l == l_min
end %for l=1:L

grad_log_ml(1,:) = sum(Aln);
theta(1,:) = theta0;

fprintf('T = 1, theta =[%.5f, %.5f]\n', theta(1,:))
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Iterate 
for n = 1:T-1
    for l = l_min:L
        %initialize
        F_f = zeros(Nl(l),2);
        F_c = F_f;
        %%%%%%%
        if l == l_min
            i1 = n*2^l+1;
            i2 = (n+1)*2^l+1;
            y = y_l{l}(i1:i2);%data at points between x_Delta_l & x_1 
            %at level l

            %Since sigma is constant, then \hat{p} is the same as
            %\tilde{p}, which is N(X0, sig^2);
            %sample X1 from the distribution N(X0, sig^2) for all particles
            %in a column vector. 
            %Note in MATLAB, in 1d, normrnd takes the standard deviations 
            %and not the variance.
            X1 = normrnd(X0_f{l},sig);

            %Generate the Brownian increments
            Z = sqrt(Delta_l(l)) * randn(Nl(l),2^l-1);

            %call the function "solve_Bridge_sde" to return the 
            %unnormalized weight $\tilde{G}_{k,\theta}^{l}$ in Eq. 4.16 
            %for all particles
            [~, G] = solve_Bridge_sde(l, Nl(l), X0_f{l}, X1, Z, y,...
                            sig, theta(n,:), kappa);
                        
            %update the weight on level l = l_min
            weight_f{l} = weight_f{l} .* G;
            %normalize
            weight_f{l} = weight_f{l}/sum(weight_f{l});

            %calculate F
            for i = 1 : Nl(l)
            %Given the Brownian increments associated with particle i and
            %X1 of particle i, call the function "solve_Bridge_sde_1" to 
            %return the particles paths Xi (a matrix of size Nl(l) x 2^l) 
            %with Xi(:,2^l) = X1(i) * ones(Nl(l),1). 
            %And it will also return the unnormalized weight 
            %$\tilde{G}_{k,\theta}^{l}$ in Eq. 4.15 for all particles.
                [Xi, Gb] = solve_bridge_sde_1(l, Nl(l), X0_f{l},...
                                Z(i,:), X1(i), y, sig,theta(n,:), kappa);

                Lambda = calc_Lambda(l,Nl(l), X0_f{l}, Xi,y,...
                                    sig, theta(n,:), kappa);
                %$\hat{p}_\theta$ vectorized
                pHat = normpdf(X1(i),X0_f{l},sig); 
                %which is the same as $\tilde{p}$ because sigma is constant
                Gb_timesP = Gb .* pHat;
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%    
                ssum = sum(Gb_timesP);

                F_f(i,:) = sum(Gb_timesP .* (Lambda + F_old_f{l})) / ssum;
            end
            
            %estimate the score function at level l=l_min on the time
            %interval n
            Aln(l,:) = sum(weight_f{l} .* F_f);
            
            %****************
            %Set things for the next iteration on the same level and 
            %resample if neccessary
            X0_f{l} = X1; 
            ESS = (sum(weight_f{l}.^2))^(-1);
            if ESS <= resamp_coef * Nl(l)
                In = randsample(Nl(l), Nl(l), true, weight_f{l});
                X0_f{l} = X0_f{l}(In); %resample
                F_f = F_f(In,:); %reorder accordingly
                weight_f{l} = ones(Nl(l),1)/Nl(l); %set weights = 1/Nl(l)
            end
            %Save F_f to be used in later iterations on the same level
            F_old_f{l} = F_f;

        else %l >= l_min+1
            i1 = n*2^l+1;
            i2 = (n+1)*2^l+1;
            j1 = n*2^(l-1)+1;
            j2 = (n+1)*2^(l-1)+1;
            
            %data at points between x_Delta_l & x_1 at level l 
            y_f = y_l{l}(i1:i2);
            %data at points between x_Delta_{l-1} & x_1 at level l-1
            y_c = y_l{l-1}(j1:j2);

            %sample X1_f and X1_c from the Thorisson coupling of
            %$\hat{p}(X0_f, X1_f)$ and $\hat{p}(X0_c, X1_c)$. That is from
            %the normal distributions N(X0_f, sig^2) & N(X0_c, sig^2) for
            %all particles

            [X1_f, X1_c] = Thorisson_coupling(Nl(l),X0_f{l},X0_c{l},sig);

            %Generate the Brownian increments on both levels
            Z_f = sqrt(Delta_l(l)) * randn(Nl(l),2^l-1);
            len = 2^(l-1)-1;
            Z_c = zeros(Nl(l),len);
            %Note: last increment in Z_f is not needed in Z_c
            for II = 1 : len
                for m = 1:2
                    k=2*(II-1)+m;
                    Z_c(:,II) = Z_c(:,II) + Z_f(:,k);
                end    
            end

            %call the function "solve_Bridge_sde" to return the 
            %unnormalized weight $\tilde{G}_{k,\theta}^{l}$ in Eq. 4.16 
            %for all particles on level l
            [~, G_f] = solve_Bridge_sde(l, Nl(l), X0_f{l}, X1_f, Z_f,...
                                        y_f, sig, theta(n,:), kappa);


            weight_f{l} = weight_f{l} .* G_f;
            weight_f{l}= weight_f{l}/ sum(weight_f{l});

            %call the function "solve_Bridge_sde" to return the 
            %unnormalized weight $\tilde{G}_{k,\theta}^{l}$ in Eq. 4.16 
            %for all particles on level l-1
            [~, G_c] = solve_Bridge_sde(l-1, Nl(l), X0_c{l}, X1_c, Z_c,...
                        y_c, sig, theta(n,:), kappa);


            weight_c{l} = weight_c{l} .* G_c;
            weight_c{l} = weight_c{l}/ sum(weight_c{l});


            %calculate F_f on level l and F_c on level l-1
            for i = 1 : Nl(l)
            %%%%%%% Vectorize Lambda and the density %%%%%%%%%%%%%%%%%%
                %Solve the diffusion bridge on the finer level l
                %One can use the same Brownian increments that were used
                %outside this loop or generate them here for each particle
                %i like: Z_f = sqrt(Delta_l(l))*randn(1,2^l-1) and instead
                %of having Z_f(i,:) in the inputs of the function below, 
                %use Z_f. Calculate Z_c from Z_f as we did before...
                [Xi, Gb] = solve_bridge_sde_1(l, Nl(l), X0_f{l}, ...
                            Z_f(i,:), X1_f(i), y_f, sig,theta(n,:), kappa);

                Lambda_f = calc_Lambda(l, Nl(l), X0_f{l}, Xi, y_f, ...
                                        sig, theta(n,:),kappa);
                pHat = normpdf(X1_f(i),X0_f{l},sig); 
                Gb_timesP = Gb .* pHat;

                ssum = sum(Gb_timesP);

                F_f(i,:)=sum(Gb_timesP .* (Lambda_f + F_old_f{l}))/ssum;
                %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                %Solve the diffusion bridge on the coarser level l-1
                [Xi, Gb] = solve_bridge_sde_1(l-1, Nl(l), X0_c{l}, ...
                            Z_c(i,:), X1_c(i), y_c, sig,theta(n,:), kappa);

                Lambda_c = calc_Lambda(l-1,Nl(l), X0_c{l}, Xi, y_c, ...
                                        sig, theta(n,:), kappa);
                pHat = normpdf(X1_c(i),X0_c{l},sig);                    
                Gb_timesP = Gb .* pHat;

                ssum = sum(Gb_timesP);

                F_c(i,:)=sum(Gb_timesP .* (Lambda_c + F_old_c{l}))/ssum;
            end

            %estimate score function 
            Aln(l,:) = sum(weight_f{l} .* F_f) - sum(weight_c{l} .* F_c);

            %****************
            %Set things for the next iteration on the same level and 
            %resample if neccessary
            X0_f{l} = X1_f;
            X0_c{l} = X1_c; 

            ESS=min((sum(weight_f{l}.^2))^(-1),(sum(weight_c{l}.^2))^(-1));
            if ESS <= resamp_coef * Nl(l)    
                [In_f, In_c] = coupled_resampling(Nl(l), weight_f{l},...
                                    weight_c{l});

                X0_f{l} = X0_f{l}(In_f); %resample
                X0_c{l} = X0_c{l}(In_c); %resample

                F_f = F_f(In_f,:); %reorder accordingly
                F_c = F_c(In_c,:); %reorder accordingly
                %set weights = 1/Nl(l)
                weight_f{l} = ones(Nl(l),1)/Nl(l);
                weight_c{l} = weight_f{l};
            end

            F_old_f{l} = F_f;
            F_old_c{l} = F_c;

        end %if l==l_min
    end %for l = l_min:L

    grad_log_ml(n+1,:) = sum(Aln);
    
    diff_grad = grad_log_ml(n+1,:) - grad_log_ml(n,:);

    %update theta
    step = (n+1)^stepsize_power;
    theta(n+1,:) = theta(n,:) + step * diff_grad;
          
    fprintf('T = %d, theta = [%.5f, %.5f]\n', n+1, theta(n+1,:))

end %n=1:T-1


%% Plotting
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

function Lambda = calc_Lambda(L,N,X0,X,y,sig,theta,kappa)
    p = 2^L;
    Delta = 1/p;
    
    Lambda = zeros(N,2);

    b = theta(1) * X0;
    Lambda(:,1) = sig^(-2) * X0 .* (X(:,1)- X0 - b * Delta);
    kappa_minus_x = kappa -X0;
    Lambda(:,2) = kappa_minus_x * (y(2)-y(1)) - theta(2) * Delta ...
                        * kappa_minus_x.^2;
    for j = 1:p-1
        b = theta(1) * X(:,j);
        Lambda(:,1) = Lambda(:,1) + sig^(-2) * X(:,j) .* ...
                        (X(:,j+1)- X(:,j) - b * Delta);
        kappa_minus_x = kappa -X(:,j);
        Lambda(:,2) = Lambda(:,2) + kappa_minus_x * (y(j+2) - y(j+1)) - ...
                        theta(2) * Delta * kappa_minus_x.^2;
    end
end



function [X, G] = solve_Bridge_sde(l,N,X0,X1,Z,y, sig,theta,kappa)  
    p = 2^l;
    Delta = 1/p;

    L = zeros(N,p);
    % \Psi = sum(L * \Delta_l), where L(:,1) is at X0 and L(:,p) is
    % at X_{1-\Delta_l}
    J = zeros(N,p); 
    %J will be summed at the end. Note that J(:,1) is at X_{\Delta_l} and
    %J(:,p) is at X1. 
    X = zeros(N,p); 
    %Euler solution of the Bridge SDE. X(:,1) is X_{\Delta_l} and X(:,p)
    % = X1
    
    kappa_minus_x = kappa-X0;
    J(:,1) = theta(2) * kappa_minus_x * (y(2)-y(1)) - 0.5 * Delta * ...
                    theta(2)^2 *kappa_minus_x.^2;
    
    temp = X1-X0; 
    b = theta(1) * X0; %b_\theta(X_t)
    
    %Calculate L at X0:
    L(:,1) = b .* temp/sig^2; %Delta_l will be multiplied at the end
    % b_\theta(X0) * grad (\log \tilde{p}(Xt;X1)) at t = 0.
    % Notice that  grad (\log \tilde{p}(Xt;X1) ) = (X1 - Xt)/(1-t)/sig^2, 
    % but t here is 0
    b_circ = b + temp; %b_\theta^\circ(X0)
    X(:,1) = X0 + b_circ * Delta + sig * Z(:,1);
    kappa_minus_x = kappa-X(:,1);
    J(:,2) = theta(2)* kappa_minus_x * (y(3)-y(2)) - ...
                0.5 * Delta * theta(2)^2 * kappa_minus_x.^2; 
    %Note A.^b means each element in A is raised to the power b
    
    for i = 1:p-2
        t = i * Delta;
        temp = (X1 - X(:,i))/(1-t);
        b = theta(1) * X(:,i);
        b_circ = b + temp;
        L(:,i+1) = b .* temp/sig^2; 
        kappa_minus_x = kappa-X(:,i+1);
        X(:,i+1) = X(:,i) + b_circ * Delta + sig * Z(:,i+1);
        J(:,i+2) = theta(2)* kappa_minus_x * (y(i+3)-y(i+2)) - ...
                     0.5 * Delta * theta(2)^2 * kappa_minus_x.^2;
    end
    
    %Set the last column in X to be X1
    X(:,p) = X1;
    %Note that t = (p-1)*Delta, hence 1-t = Delta
    temp = (X1 - X(:,p-1))/Delta; 
    b = theta(1) * X(:,p-1); %b_\theta(X_{1-\Delta_l})
    L(:,p) = b .* temp/sig^2; %G(X_{1-\Delta_l})

    G = exp(sum(J,2)+sum(L,2)*Delta);
    %Notice that sum(A,2) returns a column where each element in that
    %column is equal to the sum of corrosponding elements of all A's
    %columns
    %Notice that in the constant sigma case, we don't need the extra term
    %log (\tilde{p}/\hat{p}) because they are equal.
end

function [Xi, Gb] = solve_bridge_sde_1(l,N,X0,Z_i,X1i,y, sig, theta,kappa)

    p = 2^l;
    Delta = 1/p;

        
    Xi = zeros(N,p);
    L = zeros(N,p);
    % \Psi = sum(L * \Delta_l), where L(:,1) is at X0 and L(:,p) is at X_{1-\Delta_l}
    J = zeros(N,p);

    kappa_minus_x = kappa - X0;
    J(:,1) = theta(2) * kappa_minus_x * (y(2)-y(1)) - 0.5 * Delta * ...
                        theta(2)^2 *kappa_minus_x.^2;
    
    temp = X1i - X0; 
    b = theta(1) * X0; %b_\theta(X_t)    
    
    L(:,1) = b .* temp/sig^2;
    
    b_circ = b + temp; %b_\theta^\circ(X0)
    Xi(:,1) = X0 + b_circ * Delta + sig * Z_i(1);
    kappa_minus_x = kappa - Xi(:,1);
    J(:,2) = theta(2)* kappa_minus_x * (y(3)-y(2)) - ...
                     0.5 * Delta * theta(2)^2 * kappa_minus_x.^2;
    for j = 1:p-2
        t = j * Delta;
        temp = (X1i- Xi(:,j))/(1-t);
        b = theta(1) * Xi(:,j);
        L(:,j+1) = b .* temp/sig^2;
        b_circ = b + temp;
        Xi(:,j+1) = Xi(:,j) + b_circ * Delta + sig * Z_i(j+1);
        kappa_minus_x = kappa - Xi(:,j+1);
        J(:,j+2) = theta(2)* kappa_minus_x * (y(j+3)-y(j+2)) - ...
                     0.5 * Delta * theta(2)^2 * kappa_minus_x.^2;
    end
    
    Xi(:,p) = X1i * ones(N,1);
    %Note that t = (p-1)*Delta, hence 1-t = Delta
    temp = (X1i - Xi(:,p-1))/Delta;
    b = theta(1) * Xi(:,p-1); %b_\theta(X_{1-\Delta_l})
    L(:,p) = b .* temp/sig^2; %G(X_{1-\Delta_l})
    
    Gb = exp(sum(J,2)+sum(L,2)*Delta);
end



function [In1, In2] = coupled_resampling(N, w1, w2)
%% Coupled resampling

alphan = sum(min(w1,w2));

r = rand; %uniform random number in (0,1)
if r < alphan % with probability alphan, do the following:
    prob = min(w1, w2)/alphan;
    In1 = randsample(N,N,true,prob);
    In2 = In1;
else % with probability 1-alphan, do the following:
    prob = (w1 - min(w1,w2))/(1-alphan);
    In1 = randsample(N,N,true,prob);
    prob = (w2 - min(w1,w2))/(1-alphan);
    In2 = randsample(N,N,true,prob);
end

end


function [X1_f, X1_c] = Thorisson_coupling(N, X0_f, X0_c, sig)

% the method of Thorisson for sampling a maximal coupling associated to two
% probability densities $p$ and $f$ (with the same support) is as follows
% (below $\mathcal{U}_{[a,b]}$ is the one-dimensional uniform distribution
% on $[a,b]$, $a<b$):
% 
% 1) Sample $X$ from $p$. Sample $U$ from $\mathcal{U}_{[0,p(X)]}$. 
%       If $U<f(X)$ return (X,X), otherwise go to 2).
% 
% 2) Sample $Y$ from $f$. Sample $W$ from $\mathcal{U}_{[0,f(Y)]}$. 
%       If $W>p(Y)$ return (X,Y), otherwise start 2) again.
% 
% This is a rejection sampler, but note that the expected number of 
% proposals is at most 2. Note that if $f=p$ then one always returns $X=Y$.


    X1_f = zeros(N,1);
    X1_c = X1_f;
    t = 0;
    
    for i = 1:N
        X = normrnd(X0_f(i),sig); %X sampled from p
        a = normpdf(X,X0_f(i),sig); % = p(X)
        U = a  * rand; %uniform number on [0,a] = [0,\tilde{p}(X0_f(i),.)]
        if U < normpdf(X,X0_c(i),sig) %if U < f(X)
            X1_f(i) = X;
            X1_c(i) = X;
        else
            while t == 0
                Y = normrnd(X0_c(i),sig); % Y sampled from f
                a = normpdf(Y,X0_c(i),sig); % f(Y)
                W = a * rand; %uniform number on [0,a] = [0,\tilde{p}(X0_c(i),.)]
                if W > normpdf(Y,X0_f(i),sig) %if W > p(Y)
                    X1_f(i) = X;
                    X1_c(i) = Y;
                    t = 1;
                end
            end
        end
        
    end

end
