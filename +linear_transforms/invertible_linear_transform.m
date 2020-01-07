%
% superclass for all invertible linear transforms
%
% author: Martin F. Schiffner
% date: 2016-08-12
% modified: 2019-12-16
%
classdef (Abstract) invertible_linear_transform < linear_transforms.linear_transform

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% methods
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    methods

        %------------------------------------------------------------------
        % constructor
        %------------------------------------------------------------------
        function objects = invertible_linear_transform( N_points )

            %--------------------------------------------------------------
            % 1.) constructor of superclass
            %--------------------------------------------------------------
            % construct square-shaped linear transforms
            objects@linear_transforms.linear_transform( N_points, N_points );

        end % function objects = invertible_linear_transform( N_points )

        %------------------------------------------------------------------
        % inverse transform
        %------------------------------------------------------------------
% TODO: make method abstract
        function y = inverse_transform( LTs, x )

        end % function y = inverse_transform( LTs, x )

    end % methods

end % classdef (Abstract) invertible_linear_transform < linear_transforms.linear_transform
