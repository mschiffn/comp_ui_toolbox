%
% superclass for physical quantities with the unit meter per Volt second
%
% author: Martin F. Schiffner
% date: 2019-05-04
% modified: 2019-05-04
%
classdef meter_per_volt_second < physical_values.physical_quantity_derived

	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	%% methods
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    methods

        %------------------------------------------------------------------
        % constructor
        %------------------------------------------------------------------
        function objects = meter_per_volt_second( varargin )

            %--------------------------------------------------------------
            % 1.) constructor of superclass
            %--------------------------------------------------------------
            exponents = zeros( 1, 8 );
            exponents( 1 ) = -1;	% length
            exponents( 2 ) = -1;	% mass
            exponents( 3 ) = 2;     % time
            exponents( 4 ) = 1;     % electric current
            objects@physical_values.physical_quantity_derived( exponents, varargin{ : } );

        end % function objects = meter_per_volt_second( varargin )

	end % methods

end % classdef meter_per_volt_second < physical_values.physical_quantity_derived
