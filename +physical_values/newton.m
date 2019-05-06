%
% superclass for physical quantities with the unit Newton
%
% author: Martin F. Schiffner
% date: 2019-05-04
% modified: 2019-05-04
%
classdef newton < physical_values.force

	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	%% methods
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    methods

        %------------------------------------------------------------------
        % constructor
        %------------------------------------------------------------------
        function objects = newton( varargin )

            %--------------------------------------------------------------
            % 1.) constructor of superclass
            %--------------------------------------------------------------
            objects@physical_values.force( varargin{ : } );

        end % function objects = newton( varargin )

	end % methods

end % classdef newton < physical_values.force
