%
% superclass for physical quantities with the unit ° Celsius
%
% author: Martin F. Schiffner
% date: 2020-01-14
% modified: 2020-01-14
%
classdef degree_celsius < physical_values.temperature

	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	% methods
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    methods

        %------------------------------------------------------------------
        % constructor
        %------------------------------------------------------------------
        function objects = degree_celsius( varargin )

            %--------------------------------------------------------------
            % 1.) constructor of superclass
            %--------------------------------------------------------------
            objects@physical_values.temperature( varargin{ : } );

        end % function objects = degree_celsius( varargin )

	end % methods

end % classdef degree_celsius < physical_values.temperature
