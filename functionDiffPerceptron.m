function diff_perceptron = diffy(s)
    
    diff_perceptron = exp(-s) ./ ((1 + exp(-s)).^2);
    % differentiated perceptron
    
    
end
