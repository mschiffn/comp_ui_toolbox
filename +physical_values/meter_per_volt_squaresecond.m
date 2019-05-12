%
% superclass for physical quantities with the unit meter per Volt squaresecond
%
% author: Martin F. Schiffner
% date: 2019-05-05
% modified: 2019-05-05
%
classdef meter_per_volt_squaresecond < physical_values.physical_quantity_derived

	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	%% methods
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    methods

        %------------------------------------------------------------------
        % constructor
        %------------------------------------------------------------------
        function objects = meter_per_volt_squaresecond( varargin )

            %--------------------------------------------------------------
            % 1.) constructor of superclass
            %--------------------------------------------------------------
            exponents = zeros( 1, 8 );
            exponents( 1 ) = -1;	% length
            exponents( 2 ) = -1;	% mass
            exponents( 3 ) = 1;     % time
            exponents( 4 ) = 1;     % electric current
            objects@physical_values.physical_quantity_derived( exponents, varargin{ : } );

        end % function objects = meter_per_volt_squaresecond( varargin )

	end % methods

end % classdef meter_per_volt_squaresecond < physical_values.physical_quantity_derived
