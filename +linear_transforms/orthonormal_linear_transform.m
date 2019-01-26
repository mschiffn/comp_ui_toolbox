%
% superclass for all orthonormal linear transforms
% author: Martin Schiffner
% date: 2016-08-13
% 
classdef orthonormal_linear_transform < linear_transforms.invertible_linear_transform

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % properties
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	properties (SetAccess = private)
    end % properties

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % methods
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    methods

        %------------------------------------------------------------------
        % constructor
        %------------------------------------------------------------------
        function LT = orthonormal_linear_transform( N_lattice, str_name )

            % constructor of superclass
            LT@linear_transforms.invertible_linear_transform( N_lattice, str_name );
        end

        %------------------------------------------------------------------
        % overload method: inverse transform
        %------------------------------------------------------------------
        function y = inverse_transform( LT, x )

            % orthonormal transform : inverse transform is adjoint transform
            y = LT.adjoint_transform( x );
        end

    end % methods

end % classdef orthonormal_linear_transform < linear_transforms.invertible_linear_transform