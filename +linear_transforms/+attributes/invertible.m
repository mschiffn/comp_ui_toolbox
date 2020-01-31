%
% superclass for all invertible linear transforms
%
% author: Martin F. Schiffner
% date: 2016-08-12
% modified: 2019-12-16
%
classdef (Abstract) invertible

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% methods
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    methods

        %------------------------------------------------------------------
        % constructor
        %------------------------------------------------------------------
        function objects = invertible( N_points )

            %--------------------------------------------------------------
            % 1.) constructor of superclass
            %--------------------------------------------------------------
            % construct square-shaped linear transforms
            objects@linear_transforms.linear_transform( N_points, N_points );

        end % function objects = invertible( N_points )

	end % methods

	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	%% methods (Abstract)
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	methods (Abstract)

        %------------------------------------------------------------------
        % inverse transform
        %------------------------------------------------------------------
        y = inverse_transform( LTs, x )

	end % methods (Abstract)

end % classdef (Abstract) invertible
