%
% superclass for all unity per meters
%
% author: Martin F. Schiffner
% date: 2019-04-07
% modified: 2019-04-07
%
classdef unity_per_meter < physical_values.physical_quantity_derived

	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	% methods
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    methods

        %------------------------------------------------------------------
        % constructor
        %------------------------------------------------------------------
        function objects = unity_per_meter( varargin )

            %--------------------------------------------------------------
            % 1.) constructor of superclass
            %--------------------------------------------------------------
            exponents = zeros( 1, 8 );
            exponents( 1 ) = -1;
            objects@physical_values.physical_quantity_derived( exponents, varargin{ : } );

        end

	end % methods

end % classdef unity_per_meter < physical_values.physical_quantity_derived
