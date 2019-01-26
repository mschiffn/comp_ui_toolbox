%
% compute identity transform for various options
% author: Martin Schiffner
% date: 2016-08-13
%
classdef identity < linear_transforms.orthonormal_linear_transform

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % properties
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	properties (SetAccess = private)
        N_lattice_axis
    end % properties

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % methods
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    methods

        %------------------------------------------------------------------
        % constructor
        %------------------------------------------------------------------
        function LT_identity = identity( N_lattice_axis )

            % total number of lattice points
            N_lattice = N_lattice_axis(1) * N_lattice_axis(2);

            % constructor of superclass
            LT_identity@linear_transforms.orthonormal_linear_transform( N_lattice, 'none' );

            % internal properties
            LT_identity.N_lattice_axis = N_lattice_axis;
        end

        %------------------------------------------------------------------
        % overload method: forward transform
        %------------------------------------------------------------------
        function y = forward_transform( LT_identity, x )

            y = x;
        end

        %------------------------------------------------------------------
        % overload method: adjoint transform
        %------------------------------------------------------------------
        function y = adjoint_transform( LT_identity, x )

            y = x;
        end

    end % methods

end % classdef identity < linear_transforms.orthonormal_linear_transform