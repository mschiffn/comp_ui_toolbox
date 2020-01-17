%
% superclass for physical quantities with the unit kelvin
%
% author: Martin F. Schiffner
% date: 2020-01-14
% modified: 2020-01-14
%
classdef kelvin < physical_values.temperature

	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	% methods
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    methods

        %------------------------------------------------------------------
        % constructor
        %------------------------------------------------------------------
        function objects = kelvin( varargin )

            %--------------------------------------------------------------
            % 1.) constructor of superclass
            %--------------------------------------------------------------
            objects@physical_values.temperature( varargin{ : } );

        end % function objects = kelvin( varargin )

	end % methods

end % classdef kelvin < physical_values.temperature
