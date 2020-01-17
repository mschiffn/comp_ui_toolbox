%
% superclass for all temperatures
%
% author: Martin F. Schiffner
% date: 2020-01-14
% modified: 2020-01-14
%
classdef temperature < physical_values.physical_quantity_base

	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	% methods
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    methods

        %------------------------------------------------------------------
        % constructor
        %------------------------------------------------------------------
        function objects = temperature( varargin )

            %--------------------------------------------------------------
            % 1.) constructor of superclass
            %--------------------------------------------------------------
            objects@physical_values.physical_quantity_base( 5, varargin{ : } );

        end % function objects = temperature( varargin )

	end % methods

end % classdef temperature < physical_values.physical_quantity_base
