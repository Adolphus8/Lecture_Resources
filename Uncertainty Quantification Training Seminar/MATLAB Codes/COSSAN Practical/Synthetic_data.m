%% Synthetic data of 15 measurements of Force and Displacements:

displacement = [0.0259; 0.0276; 0.0295; 0.0367; 0.0491;...
    0.0528; 0.0579; 0.0680; 0.0688; 0.0743; 0.0748; 0.0774;...
    0.0775; 0.0779; 0.0782];

force = [-6.13; -5.77; -6.71; -10.86; -12.63; -13.17; -13.82; -18.68;...
    -18.32; -19.68; -18.26; -20.67; -18.74; -20.00; -19.85];

k = 263; % True value of stiffness [N/m]

model_output = - k .* displacement;

%% To plot the scatter plot of the measurements:

figure;
x_err = 0.003.*ones(15,1);
y_err = ones(15,1);
box on
errorbar(displacement, force, y_err, y_err, x_err, x_err,'o','Linewidth',0.8);
hold on
grid on
scatter(displacement, model_output,'Linewidth',2,'Marker',"+");
xlim([0.02 0.09])
ylim([-24 -4])
xlabel('Displacement [m]','FontSize',18)
ylabel('Force [N]','FontSize',18)
legend('Noisy data','Ideal measurements','Linewidth',2)

%% Least-squares analysis for k:

% Calculating the Least squares of k:
k_leastsquare = - displacement\force;

% Calculating the Least squares error of k:
errors_leastSquare = ((k_leastsquare - k)/k)*100;

% Print and display the results for the Least squares error of k:
sprintf('Error least square: %f%%',errors_leastSquare)

%% To plot the scatter plot of the updated model with measurements:

model_output_updated = - k_leastsquare .* displacement;

figure;
x_err = 0.003.*ones(15,1);
y_err = ones(15,1);
box on
hold on
grid on
errorbar(displacement, force, y_err, y_err, x_err, x_err,'o','Linewidth',0.8);
plot(displacement, model_output, '--k');
plot(displacement, model_output_updated, 'r');
xlim([0.02 0.09])
ylim([-24 -4])
xlabel('Displacement [m]','FontSize',18)
ylabel('Force [N]','FontSize',18)
legend('Noisy data', 'True model', 'Updated model', 'Linewidth',2, 'Fontsize', 15)
hold off
