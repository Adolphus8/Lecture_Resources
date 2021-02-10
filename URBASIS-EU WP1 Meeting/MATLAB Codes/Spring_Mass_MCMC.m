%% The Spring-Mass System:
%
% Author: Adolphus Lye
% Email: adolphus.lye@liverpool.ac.uk
%
% This example presents a simple 1DOF Spring-Mass system whose spring
% stiffness obery the Hooke's Law:
%
% F = - k . d
% where F is the force acting on the spring due to its displacement from
% its rest mass, d is the displacement length of the spring, and k is the
% spring stiffness.
%
% The value of k will be inferred from 15 measurements of F on the spring.
% Here, the spring stiffness has a true value of: k = 263 N/m 
%
% MCMC, in the form of Metropolis-Hastings, will be used to infer k.
%
%% Define the key parameters:

k_true = 263;                     % True value of stiffness [N/m]
Nobs = 10;                        % Number of measurements of F 
disp = linspace(0.01,0.08,Nobs)'; % Displacement lengths used to obtain measurements of F [m]
sigma_f = 1;                      % Standard deviation of the Gaussian measurement noise [N]

%% Define the Hooke's Law model:

force_model = @(k,d) - k .* d;     % Hooke's Law model

%% Define the synthetic measurements:

measurements = force_model(k_true,disp) + sigma_f .* randn(Nobs,1);

% Plot the measurements obtained in a graph:
figure;
hold on; box on; grid on; 
plot(disp, force_model(k_true,disp), 'k --', 'linewidth', 1)
scatter(disp, measurements, 15, 'r', 'filled')
xlabel('$d$ $[m]$', 'Interpreter', 'latex')
ylabel('$F$ $[N]$', 'Interpreter', 'latex')
xlim([0.01, 0.08])
legend('True model', 'Measurements', 'Linewidth', 2)
set(gca, 'Fontsize', 18)

% Least-squares estimate of k:
k_lse = - disp \ measurements; 
fprintf('The least-squares estimate of k is: %f \n',k_lse)

%% Bayesian Model Updating set-up:

% Define the bounds of the Uniform prior:
lowerbound = 1; upperbound = 1000;

% Define the prior:
prior_pdf = @(x) unifpdf(x, lowerbound, upperbound);
prior_rnd = @(N) unifrnd(lowerbound, upperbound, N, 1);

% Define the log-likelihood function:
logL = @(x) - 0.5 .* (1/sigma_f)^2 * sum((measurements - force_model(x,disp)).^2);

% Define the log-posterior:
log_posterior = @(x) log(prior_pdf(x)) + logL(x);

%% Metropolis-Hastings sampler set-up:

% Define the proposal distribution:
sigma_prop = 30;  % The standard deviation of the proposal distribution / tuning parameter of sampler
proppdf = @(CandidateSample,CurrentSample) normpdf(CandidateSample, CurrentSample, sigma_prop); 
proprnd = @(CurrentSample) normrnd(CurrentSample, sigma_prop); 

% Define key parameters:
Nsamples = 1000;      % No. of samples to be obtained from the posterior
burnin = 50;          % Burn in length of the Markov chain
start = prior_rnd(1); % Define the random starting sample of the Markov chain

% Initiate the MCMC sampler:
tic; 
[samples_mcmc,accept_mcmc] = mhsample(start, Nsamples, 'logpdf', log_posterior, ...
'proppdf', proppdf, 'proprnd', proprnd, 'symmetric', 1, 'burnin', burnin);
timeMCMC = toc; % To stop the timer.

fprintf('The acceptance level of the MCMC sampler is %d. \n',accept_mcmc)
fprintf('Time elapsed for the MH sampler is: %f \n',timeMCMC)

%% Analyze the results:

fprintf('The mean value of k estimate is: %f \n', mean(samples_mcmc))
fprintf('The standard deviation of k estimate is: %f \n', std(samples_mcmc))

figure;
subplot(1,2,1)
hold on; grid on; box on;
histogram(samples_mcmc, 50);
xline(k_lse, 'm', 'linewidth', 1.5)
xline(k_true, 'k', 'linewidth', 1.5)
legend('MH samples','Least-squares','True value', 'Linewidth', 2)
xlabel('$k$ $[N/m]$', 'Interpreter', 'latex')
ylabel('Count')
set(gca, 'Fontsize', 18)

subplot(1,2,2)
hold on; grid on; box on;
for idx = 1:Nsamples
plot(disp, force_model(samples_mcmc(idx,:),disp), 'color', '#C0C0C0',...
    'linewidth', 1, 'handlevisibility','off')
end
plot(disp, force_model(k_true,disp), 'k--', 'linewidth', 1)
plot(disp, force_model(k_lse,disp), 'm--', 'linewidth', 1)
scatter(disp, measurements, 15, 'r', 'filled')
legend('True model', 'Least-squares model', 'Measurements', 'Linewidth', 2)
xlabel('$d$ $[m]$', 'Interpreter', 'latex')
ylabel('$F$ $[N]$', 'Interpreter', 'latex')
xlim([0.01, 0.08])
set(gca, 'Fontsize', 18)

%% Save data

save('Spring_Mass_MCMC')