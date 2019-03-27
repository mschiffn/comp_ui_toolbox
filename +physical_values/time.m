%
% superclass for all times
%
% author: Martin F. Schiffner
% date: 2019-01-14
% modified: 2019-03-27
%
classdef time < physical_values.physical_quantity_base

	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	% methods
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    methods

        %------------------------------------------------------------------
        % constructor
        %------------------------------------------------------------------
        function objects = time( varargin )

            %--------------------------------------------------------------
            % 1.) constructor of superclass
            %--------------------------------------------------------------
            objects@physical_values.physical_quantity_base( 3, varargin{ : } );

        end % function objects = time( values )

	end % methods

end % classdef time < physical_values.physical_quantity_base
