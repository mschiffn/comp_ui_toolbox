%
% superclass for all voltages
%
% author: Martin F. Schiffner
% date: 2019-01-15
% modified: 2019-03-27
%
classdef voltage < physical_values.physical_quantity_derived

	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	% methods
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    methods

        %------------------------------------------------------------------
        % constructor
        %------------------------------------------------------------------
        function objects = voltage( varargin )

            %--------------------------------------------------------------
            % 1.) constructor of superclass
            %--------------------------------------------------------------
            exponents = zeros( 1, 8 );
            exponents( 1 ) = 2;
            exponents( 2 ) = 1;
            exponents( 3 ) = -3;
            exponents( 4 ) = -1;
            objects@physical_values.physical_quantity_derived( exponents, varargin{ : } );

        end % function objects = voltage( varargin )

	end % methods

end % classdef voltage < physical_values.physical_quantity_derived
