%% The 2DOF Shear Building:
%
% Author: Adolphus Lye
% Email: adolphus.lye@liverpool.ac.uk
%
% This example is adopted from the literature by J. L. Beck and S. K. Au (2002):
% S. K. Au and J. L. Beck (2002). Bayesian Updating of Structural Models and 
% Reliability using Markov Chain Monte Carlo Simulation. Journal of
% Engineering Mechanics, 128(4), 380-391. doi: 10.1061/(ASCE)0733-9399(2002)128:4(380)
%
% This example presents a 2DOF Shear Building whose inter-storey stiffnesses
% are k1 and k2. The storey masses are m1 and m2 respectively.
% Here, m1 = m2 = 1.0 x 10^4 kg
%
% The stiffness matrix is defined as:
% [k1 + k2, -k2; -k2, k2]
% where the true values of k1 and k2 are: k1 = 5 x 10^3 N/m, k2 = 1.5 x 10^4 N/m
%
% The 2DOF Shear Building has 2 eigenfrequencies corresponding to its 2
% modes of vibration. 10 measurements of eigenfrequencies {f1,f2} will be
% obtained from its response dynamics and will be used to infer k1 and k2.
%
% TMCMC will be used to infer k1 and k2 from a bi-modal posterior. As a
% comparison, the MCMC sampling technique will be implemented alongside.
%
%% Define key parameters:

k1_true = 5e+03;   % Inter-storey stiffness k1 [N/m]
k2_true = 1.5e+04; % Inter-storey stiffness k2 [N/m]
Nobs = 10;         % Number of measurements of {f1,f2} 
m = 1e+04;         % Storey mass [kg]
sigma_f1 = 1;      % Standard deviation of the Gaussian measurement noise for f1 [Hz]
sigma_f2 = 0.1;    % Standard deviation of the Gaussian measurement noise for f2 [Hz]

%% Define the model for the eigen-frequencies:

freq_model_1 = @(x) 0.5.*(1/m).*((x(:,1) + 2.*x(:,2)) + (x(:,1).^2 + 4.*(x(:,2).^2)).^0.5);
freq_model_2 = @(x) 0.5.*(1/m).*((x(:,1) + 2.*x(:,2)) - (x(:,1).^2 + 4.*(x(:,2).^2)).^0.5);

%% Define the synthetic measurements:

measurements_f1 = freq_model_1([k1_true, k2_true]) + sigma_f1 .* randn(Nobs,1);
measurements_f2 = freq_model_2([k1_true, k2_true]) + sigma_f2 .* randn(Nobs,1);

% Plot the measurements obtained in a graph:
figure;
hold on; box on; grid on; 
scatter(measurements_f1, measurements_f2, 15, 'r', 'filled')
plot(freq_model_1([k1_true, k2_true]), freq_model_2([k1_true, k2_true]), 'k+', 'Linewidth', 2)
xlabel('$f_{1}$ $[Hz]$', 'Interpreter', 'latex')
ylabel('$f_{2}$ $[Hz]$', 'Interpreter', 'latex')
legend('Measurements', 'True values', 'Linewidth', 2)
set(gca, 'Fontsize', 18)

%% Bayesian Model Updating set-up:

% Define the bounds of the Uniform prior:
lowerbound = 1; upperbound = 1e+05;

% Define the prior:
prior_pdf_k1 = @(x) unifpdf(x, lowerbound, upperbound); % Prior for k1
prior_pdf_k2 = @(x) unifpdf(x, lowerbound, upperbound); % Prior for k2
prior_pdf = @(x) prior_pdf_k1(x(:,1)) .* prior_pdf_k2(x(:,2)); % Overall prior

prior_rnd = @(N) [unifrnd(lowerbound, upperbound, N, 1),...
                  unifrnd(lowerbound, upperbound, N, 1)];
              
% Define the log-likelihood function:
logL = @(x) - 0.5 .* (1/sigma_f1)^2 * sum((measurements_f1 - freq_model_1(x)).^2) + ...
            - 0.5 .* (1/sigma_f2)^2 * sum((measurements_f2 - freq_model_2(x)).^2);

% Define the log-posterior:
log_posterior = @(x) log(prior_pdf(x)) + logL(x);

%% Metropolis-Hastings sampler set-up:

% Define the proposal distribution:
cov_prop = 6e+06.*eye(2);  % The covariance matrix of the proposal distribution / tuning parameter of sampler
proppdf = @(CandidateSample,CurrentSample) mvnpdf(CandidateSample, CurrentSample, cov_prop);
proprnd = @(CurrentSample) mvnrnd(CurrentSample, cov_prop); 

% Define key parameters:
Nsamples = 1000;      % No. of samples to be obtained from the posterior
burnin = 100;         % Burn in length of the Markov chain
start = prior_rnd(1); % Define the random starting sample of the Markov chain

% Initiate the MCMC sampler:
tic; 
[samples_mcmc,accept_mcmc] = mhsample(start, Nsamples, 'logpdf', log_posterior, ...
'proppdf', proppdf, 'proprnd', proprnd, 'symmetric', 1, 'burnin', burnin);
timeMCMC = toc; % To stop the timer.

fprintf('The acceptance level of the MCMC sampler is %d. \n',accept_mcmc)
fprintf('Time elapsed for the MH sampler is: %f \n',timeMCMC)

%% TMCMC sampler set-up:

tic;
TMCMC = TMCMCsampler('nsamples', Nsamples, 'loglikelihood', logL, 'priorpdf', prior_pdf, ...
'priorrnd', prior_rnd, 'burnin', 0);
timeTMCMC = toc; % To stop the timer.

samples_tmcmc = TMCMC.samples;

fprintf('Time elapsed for the TMCMC sampler is: %f \n',timeTMCMC)

%% Analyze the results:

% Plot the evolution of the posterior:
allsamples_tmcmc = TMCMC.allsamples;
figure;
hold on; 
for idx = 1:size(allsamples_tmcmc,3)
subplot(2,4,idx)
hold on; box on; grid on;
scatter(allsamples_tmcmc(:,1,idx), allsamples_tmcmc(:,2,idx), 15, 'b', 'filled')
xlabel('$k_{1}$ $[N/m]$', 'Interpreter', 'latex')
ylabel('$k_{2}$ $[N/m]$', 'Interpreter', 'latex')
title(['j = ',num2str(idx-1)])
set(gca, 'Fontsize', 13)
xlim([0 10e+04]); ylim([0 10e+04]);
end

% Plot the posterior samples
figure;
subplot(1,2,1)
hold on; grid on; box on;
scatter(samples_tmcmc(:,1), samples_tmcmc(:,2), 15, 'b', 'filled')
plot(k1_true, k2_true, 'k+', 'Linewidth', 2)
legend('TMCMC samples','True values', 'Linewidth', 2)
xlabel('$k_{1}$ $[N/m]$', 'Interpreter', 'latex')
ylabel('$k_{2}$ $[N/m]$', 'Interpreter', 'latex')
xlim([0 5e+04]); ylim([0 2.5e+04]);
set(gca, 'Fontsize', 18)

subplot(1,2,2)
hold on; grid on; box on;
scatter(samples_mcmc(:,1), samples_mcmc(:,2), 15, 'b', 'filled')
plot(k1_true, k2_true, 'k+', 'Linewidth', 2)
legend('MH samples','True values', 'Linewidth', 2)
xlabel('$k_{1}$ $[N/m]$', 'Interpreter', 'latex')
ylabel('$k_{2}$ $[N/m]$', 'Interpreter', 'latex')
xlim([0 5e+04]); ylim([0 2.5e+04]);
set(gca, 'Fontsize', 18)

% Plot the updated model
figure;
subplot(1,2,1)
hold on; grid on; box on;
scatter(freq_model_1(samples_mcmc), freq_model_2(samples_mcmc), 15, 'b', 'filled')
scatter(measurements_f1, measurements_f2, 15, 'r', 'filled')
plot(freq_model_1([k1_true, k2_true]), freq_model_2([k1_true, k2_true]), 'k+', 'Linewidth', 2)
xlabel('$f_{1}$ $[Hz]$', 'Interpreter', 'latex')
ylabel('$f_{2}$ $[Hz]$', 'Interpreter', 'latex')
legend('MH model update', 'Measurements', 'True values', 'Linewidth', 2)
set(gca, 'Fontsize', 18)

subplot(1,2,2)
hold on; grid on; box on;
scatter(freq_model_1(samples_tmcmc), freq_model_2(samples_tmcmc), 15, 'b', 'filled')
scatter(measurements_f1, measurements_f2, 15, 'r', 'filled')
plot(freq_model_1([k1_true, k2_true]), freq_model_2([k1_true, k2_true]), 'k+', 'Linewidth', 2)
xlabel('$f_{1}$ $[Hz]$', 'Interpreter', 'latex')
ylabel('$f_{2}$ $[Hz]$', 'Interpreter', 'latex')
legend('TMCMC model update', 'Measurements', 'True values', 'Linewidth', 2)
set(gca, 'Fontsize', 18)

%% Save data

save('Shear_Building_TMCMC')