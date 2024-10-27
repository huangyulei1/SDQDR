function [X_new,V,W,H,A,C,objsum,iter,score]=SDQDR(X,C,kk,Ds,Ws,Dp,Wp,alpha,beta,theta,NIter,lamda,p)
% SDQDR: Sparse dual-graph regularized quadratic dimensionality reduction algorithm based on subspace learning.
% 
% Input:
%   X       - Data matrix (d*n). Each column vector of data is a sample vector.
%   C       - The label matrix.
%   kk      - The number of classes.
%   Ds      - The degree matrix of data space.
%   Dp      - The degree matrix of feature space.
%   Ws      - The similarity matrix of data space.
%   Wp      - The similarity matrix of feature space.
%   alpha   - The graph regularization parameter.
%   beta    - The orthogonal constraint parameter.
%   lamda   - The sparse constraint parameter.
%   theta   - The sparse constraint parameter.
%   p       - The number of featuers to be selected.
%   NIter   - The maximum number of iterations.
%
% Output:
%   X_new   - The selected subset of features (n*p).
%--------------------------------------------------------------------------

[m,~] = size(X);
I = eye(kk);
Im = ones(kk,1);
W = rand(m,kk);
H = Init_R(X,kk);
[~,c] = size(C);
A = rand(c,kk);
for iter = 1:NIter
    G = sqrt(diag(sum((C*A)'*(C*A),2)));
    W = W*pinv(G);
    H = G*H;
	% ===================== update W & Q ========================
    W = W.*(X*C*A*H' + alpha*Wp*W*H*H' + beta*W)./(W*H*A'*C'*C*A*H' + alpha*Dp*W*H*H' + beta*W*W'*W + eps);
    Hi = sqrt(sum(H.*H,2) + eps);
    d = 0.25./(Hi.^3/2);
    Q = diag(d);
	% ===================== update H & A & C========================
	H = H.*(W'*X*C*A + alpha*W'*Wp*W*H)./(W'*W*H*A'*C'*C*A + alpha*W'*Dp*W*H + theta*Im*Im' +lamda*Q*H + eps);
    A = A.*(C'*X'*W*H + alpha*C'*Ws*C*A + beta*C'*C*A)./(C'*C*A*H'*W'*W*H + alpha*C'*Ds*C*A + beta*C'*C*A*A'*C'*C*A + eps);
    C = C.*(X'*W*H*A' + alpha*Ws*C*A*A' + beta*C*A*A')./(C*A*H'*W'*W*H*A' + alpha*Ds*C*A*A' + beta*C*A*A'*C'*C*A*A' + eps);
    % ==============================================================
    obj(1,iter) = trace(X*X' - 2*W*H*A'*C'*X' + W*H*A'*C'*C*A*H'*W');
    obj(2,iter) = alpha*(trace(A'*C'*(Ds-Ws)*C*A)+trace(H'*W'*(Dp-Wp)*W*H));
    obj(3,iter) = 0.5*beta*(trace((W'*W-I)*(W'*W-I)') + trace((A'*C'*C*A-I)*(A'*C'*C*A-I)'));
    obj(4,iter) = theta*trace(Im'*H'*Im) + lamda*trace(H'*Q*H);
    objsum(iter) = sum(obj(:,iter));
    if iter > 2
        if abs(objsum(iter)-objsum(iter-1)) < 0.01
            break
        end
    end
end

V = C*A;
score = sum(H'.*H',2);
[~, idx] = sort(score,'descend');
Y = V';
X_new = Y(idx(1:p),:);
end