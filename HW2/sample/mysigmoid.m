function G = mysigmoid(U,V)
% Sigmoid kernel function with slope gamma and intercept c
display(U);
display(V);
gamma = 1;
c = -1;
G = tanh(gamma*U*V' + c);
end

