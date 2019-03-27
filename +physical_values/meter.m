%
% superclass for physical quantities with the unit meter
%
% author: Martin F. Schiffner
% date: 2019-03-26
% modified: 2019-03-27
%
classdef meter < physical_values.length

	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	% methods
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    methods

        %------------------------------------------------------------------
        % constructor
        %------------------------------------------------------------------
        function objects = meter( varargin )

            %--------------------------------------------------------------
            % 1.) constructor of superclass
            %--------------------------------------------------------------
            objects@physical_values.length( varargin{ : } );

        end % function objects = meter( varargin )

	end % methods

end % classdef meter < physical_values.length
