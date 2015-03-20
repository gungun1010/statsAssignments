function k = gaussianKernel(X,X_in)
    var = 100;
    display(X);
    display(X_in);
    k = exp(-norm(X-X_in)*norm(X-X_in)/(2*var^2));
end

