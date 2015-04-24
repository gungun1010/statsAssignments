function err = critfun(XT, YT, Xt, Yt)

% fprintf('XT = %d\n',size(XT));
% fprintf('YT = %d\n',size(YT));
% fprintf('Xt = %d\n',size(Xt));
% fprintf('Yt = %d\n',size(Yt));

theta = ((XT')*XT)^-1 * XT'*YT;

err = sum((Yt - Xt*theta).^2);