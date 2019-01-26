%
% compute effect of diagonal weighting matrix
% author: Martin Schiffner
% date: 2016-08-13
%
classdef weighting < linear_transforms.invertible_linear_transform

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % properties
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	properties (SetAccess = private)
        weights
        weights_conj
    end % properties

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % methods
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    methods

        %------------------------------------------------------------------
        % constructor
        %------------------------------------------------------------------
        function LT_weighting = weighting( weights )

            % check for zero weights
            indicator = abs( weights(:) ) < eps;
            if sum( indicator, 1 ) > 0
                fprintf('small weights detected!\n');
                return
            end

            % number of weights
            N_weights = numel( weights );

            % constructor of superclass
            LT_weighting@linear_transforms.invertible_linear_transform( N_weights, 'weighting' );

            % internal properties
            LT_weighting.weights        = weights(:);
            LT_weighting.weights_conj	= conj( LT_weighting.weights );
        end

        %------------------------------------------------------------------
        % overload method: forward transform (forward weighting)
        %------------------------------------------------------------------
        function y = forward_transform( LT_weighting, x )

            y = x(:) .* LT_weighting.weights;
        end

        %------------------------------------------------------------------
        % overload method: adjoint transform (adjoint weighting)
        %------------------------------------------------------------------
        function y = adjoint_transform( LT_weighting, x )

            y = x(:) .* LT_weighting.weights_conj;
        end

        %------------------------------------------------------------------
        % overload method: inverse transform (inverse weighting)
        %------------------------------------------------------------------
        function y = inverse_transform( LT_weighting, x )

            y = x(:) ./ LT_weighting.weights;
        end

    end % methods

end % classdef weighting < linear_transforms.invertible_linear_transform