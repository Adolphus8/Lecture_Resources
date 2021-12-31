%% Tutorial: 1DOF Linear Static Spring-Mass System
%
% Authors: Adolphus Lye & Ander Gray
%
% In this tutorial, we will be analysing a 1D Linear Static Spring system
% whose spring obeys the Hooke's Law:
%
% F = - k.d
%
% whereby F is the force acting on the spring, x is the displacement of the
% spring from its rest length, and k is the spring constant. Here, k is the
% epistemic parameter and in this tutorial, we will be using the Bayesian
% Model Updating technique to estimate this value of k.
%
% The true value of k = 263 N/m
% The displacement, d = 0.05 m
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%
opencossan.OpenCossan.getVerbosityLevel
%% Task 2: Preparation/Definition of the parameters and random variables

%% Task 2(a): Definition of the Parameters using 'Parameter' constructor from OpenCossan

% Define the true stiffness parameter, k, using the OpenCOSSAN ‘Parameter’ constructor [N/m]:
stiffness_real = opencossan.common.inputs.Parameter('value',263,'description','stiffness_real');

% Define the displacement parameter, d, using the OpenCOSSAN ‘Parameter’ constructor [m]:
displacement = opencossan.common.inputs.Parameter('value',0.05,'description','displacement');

% Define the noise random variable, NoiseStandardDeviation, using the OpenCOSSAN ‘RandomVariable’ constructor for a Normal distribution:
NoiseStandardDeviation = opencossan.common.inputs.random.NormalRandomVariable('mean',0.0,'std',1.0,'description','NoiseStandardDeviation');

% Consolidate the random data using the OpenCOSSAN `RandomVariableSet’ constructor:
Xrvset = opencossan.common.inputs.random.RandomVariableSet('members',[NoiseStandardDeviation],'names',['NoiseStandardDeviation']);

%% Task 2(b): Simulate perturbed data of F_noisy (via inputs - stiffness_real and displacement)

% Consolidate the input parameters and random variables for the “noisy" model using theOpenCOSSAN ’Input’ constructor:
XinputPert=opencossan.common.inputs.Input('Members',{stiffness_real, displacement, Xrvset},'MembersNames', {'stiffness', 'displacement', 'Xrvset'});
display(XinputPert); 

% Prepare the perturbed Model by 'Mio' constructor from OpenCossan using a matlab script that is coded 
% to handle the random vector to be added to the output displacements (to simulate synthetic data):

% Prepare the “noisy" model function handle using the OpenCOSSAN 'Mio’ constructor:
XmioPert=opencossan.workers.Mio('FunctionHandle', @model,...
    'IsFunction', true, ...
    'inputnames',{'stiffness', 'displacement', 'NoiseStandardDeviation'}, ...
    'outputnames',{'y'},...
    'Format', 'table');

% Add the 'mio' constructor to 'Evaluator' constructor from OpenCossan:
XevaluatorPert=opencossan.workers.Evaluator('CXmembers',{XmioPert},'CSmembers',{'XmioPert'});

% Defining the "noisy" Model by the 'Model' constructor from OpenCossan:
XmodelPert=opencossan.common.Model('input', XinputPert,'evaluator', XevaluatorPert);

% Prepare 'MonteCarlo' constructor from OpenCossan:
simulation_no = 500;
Xmc=opencossan.simulations.MonteCarlo('Nsamples', simulation_no);

% Perform Monte Carlo simulation using 'Apply' method:
XsyntheticData=Xmc.apply(XmodelPert);

% Display the resulting outputs:
XsyntheticData.getValues('Cnames',{'y'});

%% Task 3: To perform determinsitic analysis

% Consolidate the input parameters and random variables for the nominal model using theOpenCOSSAN ’Input’ constructor:
Xinput=opencossan.common.inputs.Input('Members',{stiffness_real displacement},'MembersNames',{'stiffness' 'displacement'});
display(Xinput);

% Prepare the nominal model function handle using the OpenCOSSAN 'Mio’ constructor:
Xmio=opencossan.workers.Mio('FunctionHandle',@model_nominal,...
    'IsFunction', true,...
    'inputnames',{'stiffness', 'displacement'}, ...
    'outputnames',{'y'},...
    'Format', 'table');

% Add the 'mio' constructor to 'Evaluator' constructor from OpenCossan:
Xevaluator=opencossan.workers.Evaluator('CXmembers',{Xmio},'CSmembers',{'Xmio'});

% Defining the nominal model by the 'Model' constructor from OpenCossan:
Xmodel1DOFModelUpdating=opencossan.common.Model('input',Xinput,'evaluator',Xevaluator);

% To execute Deterministic analysis:
Xout=deterministicAnalysis(Xmodel1DOFModelUpdating);

% To yield the actual true value of F:
ActualForce=Xout.getValues('Cnames',{'y'});          

%% Task 4: Bayesian Model Updating

%% Task 4(a): Define the Prior and the Likelihood functions

% Define key prior parameters:
lowerBound = 1;    % [N/m]
upperBound = 1000; % [N/m]

% Define the prior random variable using the OpenCOSSAN ’RandomVariable’ constructor for a Uniform distribution:
stiffness = opencossan.common.inputs.random.UniformRandomVariable('bounds',[lowerBound, upperBound],'description','Stiffness of spring');

% Consolidate the random data from the prior using the OpenCOSSAN ‘RandomVariableSet’ constructor:
Xrvset = opencossan.common.inputs.random.RandomVariableSet('members',[stiffness],'names',['stiffness']);

% Consolidate the input prior random variable for the nominal model using the OpenCOSSAN ‘Input’ constructor:
XinputBayes = opencossan.common.inputs.Input('Members',{Xrvset, displacement},'MembersNames',{'Xrvset', 'displacement'});
display(XinputBayes); 
%Sfolder = fileparts(which('Linear_Static_tutorial.m')); 

% Prepare the nominal model function handle using the OpenCOSSAN ‘Mio’ constructor:
XmioBayes = opencossan.workers.Mio('FunctionHandle', @model_nominal,...
    'IsFunction', true, ...
    'inputnames',{'stiffness' 'displacement'}, ...
    'outputnames',{'y'},...
    'Format', 'table');

% Set up the model evaluator by adding the ‘Mio’ constructor in the previous step to theOpenCOSSAN ‘Evaluator’ constructor:
XevaluatorBayes = opencossan.workers.Evaluator('CXmembers',{XmioBayes},'CSmembers',{'XmioBayes'});

% Define the nominal model using the OpenCOSSAN ‘Model’ constructor:
XmodelBayes = opencossan.common.Model('input',XinputBayes,'evaluator',XevaluatorBayes);

% Define the loglikelihood function using the OpenCossan ‘LogLikelihood’ constructor:
log_l = opencossan.inference.LogLikelihood('Model', XmodelBayes, 'Data', XsyntheticData, 'WidthFactor', [1]);

%% Task 4(b): Set up the TMCMC sampler:

% Defining the Bayesian Model Updating object:
Nsamples = 1000;

% Consolidate the prior and the loglikelihood functions using the OpenCOSSAN ‘Bayesian-ModelUpdating’ object:
Bayes = opencossan.inference.BayesianModelUpdating('LogLikelihood', log_l ,'OutputNames', ["stiffness"], 'Nsamples', Nsamples);

% Run the TMCMC sampler:
tic;
samps = applyTMCMC(Bayes);
timeTMCMC = toc;
fprintf('The computational time of the TMCMC sampler is: %4.2f s. \n', timeTMCMC)

% To yield the sample values generated using the TMCMC sampler:
samples_tmcmc = samps.getValues('Cnames',{'stiffness'});

%% Task 5: Analysis sample outputs:

% This is to plot the histogram of the posterior samples obtained via TMCMC:
figure;
hold on; box on; grid on;
nbins = 50; % Number of bins for the histogram
histogram(samples_tmcmc(:,1),nbins, 'handlevisibility', 'off'); 
xline(stiffness_real.Value, 'k--', 'linewidth', 1.5)
xlabel('$k$ $[N/m]$', 'Interpreter', 'latex'); ylabel('Count');
legend('True value = $263$ $N/m$', 'Interpreter', 'latex', 'linewidth', 2)
set(gca,'FontSize',20)

% To calculate sample mean:
tmcmc_mean_k = mean(samples_tmcmc(:,1)); 

% To calculate sample standard deviation:
tmcmc_std_k = std(samples_tmcmc(:,1));   

% To display in the Command Window the mean and standard deviation of k obtained via the TMCMC sampler:
fprintf('Estimated k via TMCMC: %4.2f N/m; with standard deviation: %4.2f N/m. \n',tmcmc_mean_k,tmcmc_std_k)

% To calculate the percentage error of the estimation:
Percentage_error_tmcmc = (tmcmc_std_k./tmcmc_mean_k)*100; 

% To display in the Command Window the percentage error of the estimation of k obtained by the TMCMC sampler:
fprintf('The percentage error of the estimation of k via TMCMC is: %3.2f%% \n', Percentage_error_tmcmc);

% To obtain synthetic data for output (Force measurement):
synthetic_force_measurement = table2array(XsyntheticData.TableValues(:,1));

% To obtain output from calibrated/updated model:
Params = XmodelBayes.Input.Parameters; 
for n = fieldnames(Params)'
    Params.(n{1}) = repmat(Params.(n{1}).Value, Nsamples, 1);
end
Tinput = struct2table(Params);
Table_theta = array2table(samples_tmcmc);
Table_theta.Properties.VariableNames = XmodelBayes.Input.RandomVariableNames;
Tinput = [Table_theta, Tinput];
Xout = XmodelBayes.apply(Tinput);
Update_output = Xout.getValues('Cnames', {'y'});

% Plotting histogram of Updated model output vs Synthetic data:
figure;
hold on; box on; grid on;
histogram(synthetic_force_measurement, 50);
histogram(Update_output, 50);
xlabel('$F$ $[N]$', 'Interpreter', 'latex'); ylabel('Count');
legend('Synthetic data', 'Updated Model', 'Fontsize', 15, 'Linewidth', 2)
set(gca,'FontSize',20)
