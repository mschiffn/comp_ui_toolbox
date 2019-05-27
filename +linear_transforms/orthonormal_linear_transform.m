%
% superclass for all orthonormal linear transforms
%
% author: Martin F. Schiffner
% date: 2016-08-13
% modified: 2019-05-27
%
classdef orthonormal_linear_transform < linear_transforms.invertible_linear_transform

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% methods
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	methods

        %------------------------------------------------------------------
        % constructor
        %------------------------------------------------------------------
        function objects = orthonormal_linear_transform( N_points )

            %--------------------------------------------------------------
            % 1.) constructor of superclass
            %--------------------------------------------------------------
            % construct invertible linear transforms
            objects@linear_transforms.invertible_linear_transform( N_points );

        end % function objects = orthonormal_linear_transform( N_points )

        %------------------------------------------------------------------
        % inverse transform (overload inverse_transform method)
        %------------------------------------------------------------------
        function y = inverse_transform( LTs, x )

            % orthonormal transform : inverse transform is adjoint transform
            y = adjoint_transform( LTs, x );

        end % function y = inverse_transform( LTs, x )

    end % methods

end % classdef orthonormal_linear_transform < linear_transforms.invertible_linear_transform
