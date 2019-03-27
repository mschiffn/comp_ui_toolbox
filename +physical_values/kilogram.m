%
% superclass for all masses with the unit kilogram
%
% author: Martin F. Schiffner
% date: 2019-03-27
% modified: 2019-03-27
%
classdef kilogram < physical_values.mass

	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	% methods
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    methods

        %------------------------------------------------------------------
        % constructor
        %------------------------------------------------------------------
        function objects = kilogram( varargin )

            %--------------------------------------------------------------
            % 1.) constructor of superclass
            %--------------------------------------------------------------
            objects@physical_values.mass( varargin{ : } );

        end % function objects = kilogram( varargin )

	end % methods

end % classdef kilogram < physical_values.mass
