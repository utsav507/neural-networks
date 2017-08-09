function y = perceptron(s)

    y = 1./(1 + exp(-s));
    
%     plot(s,y)
%     y(y >= 0.75) = 1;
%     y(y <= 0.25) = 0;
    
end

% y = zeros(size(s));
%     for counter = 1:length(s)
%         if s(counter) > 0
%             y(counter) = 1;
%         else
%             y(counter) = 0;
%         end
%     end
% end