%
% superclass for all invertible linear transforms
% author: Martin Schiffner
% date: 2016-08-12
% 
classdef invertible_linear_transform < linear_transforms.linear_transform

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
        function LT = invertible_linear_transform( N_lattice, str_name )

            % constructor of superclass
            % invertible linear transforms are square-shaped
            LT@linear_transforms.linear_transform( N_lattice, N_lattice, str_name );
        end

        %------------------------------------------------------------------
        % inverse transform
        %------------------------------------------------------------------
        function y = inverse_transform( LT, x )

        end

    end % methods

end % classdef invertible_linear_transform < linear_transforms.linear_transform