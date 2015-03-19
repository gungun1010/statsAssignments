function k = gaussianKernenl(X,X_in)
    var = 1;
    k = exp(-norm(X-X_in)*norm(X-X_in)/(2*var^2));
    display(k);
end

