%% Task 1(b): Setting up the "noisy" Hooke's Law model for the Linear Static Spring-Mass system

function [force] = model(in)

%MODEL: 1-Degree of freedom mass-spring system 

force = table();
y = zeros(height(in), 1);

for i = 1:height(in)
    y(i,:) = - in.stiffness(i) .* in.displacement(i);
end

force.y = y(:,1) + in.NoiseStandardDeviation;

end

