%
% superclass for physical quantities with the unit second
%
% author: Martin F. Schiffner
% date: 2019-03-26
% modified: 2019-03-26
%
classdef second < physical_values.time

	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	% methods
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    methods

        %------------------------------------------------------------------
        % constructor
        %------------------------------------------------------------------
        function objects = second( varargin )

            %--------------------------------------------------------------
            % 1.) constructor of superclass
            %--------------------------------------------------------------
            objects@physical_values.time( varargin{ : } );

        end % function objects = second( varargin )

	end % methods

end % classdef second < physical_values.time
