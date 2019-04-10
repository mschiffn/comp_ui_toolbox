%
% superclass for physical quantities with the unit pascal
%
% author: Martin F. Schiffner
% date: 2019-04-08
% modified: 2019-04-08
%
classdef pascal < physical_values.physical_quantity_derived

	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	% methods
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    methods

        %------------------------------------------------------------------
        % constructor
        %------------------------------------------------------------------
        function objects = pascal( varargin )

            %--------------------------------------------------------------
            % 1.) constructor of superclass
            %--------------------------------------------------------------
            exponents = zeros( 1, 8 );
            exponents( 1 ) = -1;    % length
            exponents( 2 ) = 1;     % mass
            exponents( 3 ) = -2;    % time
            objects@physical_values.physical_quantity_derived( exponents, varargin{ : } );

        end

	end % methods

end % classdef pascal < physical_values.physical_quantity_derived
