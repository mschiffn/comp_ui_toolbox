%
% superclass for physical quantities with the unit Volt per Newton second
%
% author: Martin F. Schiffner
% date: 2019-05-05
% modified: 2019-05-05
%
classdef volt_per_newton_second < physical_values.physical_quantity_derived

	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	%% methods
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    methods

        %------------------------------------------------------------------
        % constructor
        %------------------------------------------------------------------
        function objects = volt_per_newton_second( varargin )

            %--------------------------------------------------------------
            % 1.) constructor of superclass
            %--------------------------------------------------------------
            exponents = zeros( 1, 8 );
            exponents( 1 ) = 1;     % length
            exponents( 3 ) = -2;	% time
            exponents( 4 ) = -1;	% electric current
            objects@physical_values.physical_quantity_derived( exponents, varargin{ : } );

        end % function objects = volt_per_newton_second( varargin )

	end % methods

end % classdef volt_per_newton_second < physical_values.physical_quantity_derived
