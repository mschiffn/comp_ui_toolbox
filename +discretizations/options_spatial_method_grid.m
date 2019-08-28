% abstract superclass for all spatial discretization methods using grids
%
% author: Martin F. Schiffner
% date: 2019-08-20
% modified: 2019-08-20
%
classdef (Abstract) options_spatial_method_grid < discretizations.options_spatial_method

	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	%% methods
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    methods

        %------------------------------------------------------------------
        % constructor
        %------------------------------------------------------------------
        function objects = options_spatial_method_grid( size )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % superclass ensures row vector for size
            % superclass ensures nonempty positive integers for size

            %--------------------------------------------------------------
            % 2.) create options_spatial_method_grid options
            %--------------------------------------------------------------
            % constructor of superclass
            objects@discretizations.options_spatial_method( size );

        end % function objects = options_spatial_method_grid( size )

	end % methods

end % classdef (Abstract) options_spatial_method_grid < discretizations.options_spatial_method
