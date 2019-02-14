function [learned_dictionary param ] = construct_dict_comm(param,dimension,p)
%construct dictionary and save Laplacian powers
%take the reduced Laplacian
    for k = 0 : max(param.K)
        eval(['param.Laplacian_powers',num2str(p),'{k + 1} = param.Laplacian',num2str(p),'^k;']);
    end

    for i=1:param.S
        learned_dict{i} = zeros(dimension);
    end

    for k = 1 : max(param.K)+1
        for i=1:param.S
            eval(['learned_dict{i} = learned_dict{i} + param.alpha{i}(k)*param.Laplacian_powers',num2str(p),'{k};']);
        end
    end

    learned_dictionary = [learned_dict{1}];
    for i = 2: param.S
            learned_dictionary = [learned_dictionary, learned_dict{i}];
    end
end

