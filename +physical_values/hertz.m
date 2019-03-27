%
% superclass for all time frequencies in Hertz
%
% author: Martin F. Schiffner
% date: 2019-03-27
% modified: 2019-03-27
%
classdef hertz < physical_values.frequency

	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	% methods
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    methods

        %------------------------------------------------------------------
        % constructor
        %------------------------------------------------------------------
        function objects = hertz( varargin )

            %--------------------------------------------------------------
            % 1.) constructor of superclass
            %--------------------------------------------------------------
            objects@physical_values.frequency( varargin{ : } );

        end % function objects = hertz( varargin )

	end % methods

end % classdef hertz < physical_values.frequency
