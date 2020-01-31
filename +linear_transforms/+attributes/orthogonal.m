%
% superclass for all orthonormal linear transforms
%
% author: Martin F. Schiffner
% date: 2016-08-13
% modified: 2019-12-16
%
classdef (Abstract) orthogonal < linear_transforms.attributes.invertible

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% methods
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	methods

        %------------------------------------------------------------------
        % constructor
        %------------------------------------------------------------------
        function objects = orthogonal( N_points )

            %--------------------------------------------------------------
            % 1.) constructor of superclass
            %--------------------------------------------------------------
            % construct invertible linear transforms
            objects@linear_transforms.attributes.invertible( N_points );

        end % function objects = orthogonal( N_points )

        %------------------------------------------------------------------
        % inverse transform (overload inverse_transform method)
        %------------------------------------------------------------------
        function y = inverse_transform( LTs, x )

            % orthonormal transform : inverse transform is adjoint transform
            y = adjoint_transform( LTs, x );

        end % function y = inverse_transform( LTs, x )

    end % methods

end % classdef (Abstract) orthogonal < linear_transforms.attributes.invertible
