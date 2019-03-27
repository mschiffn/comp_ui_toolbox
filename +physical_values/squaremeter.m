%
% superclass for all square meters
%
% author: Martin F. Schiffner
% date: 2019-03-27
% modified: 2019-03-27
%
classdef squaremeter < physical_values.physical_quantity_derived

	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	% methods
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    methods

        %------------------------------------------------------------------
        % constructor
        %------------------------------------------------------------------
        function objects = squaremeter( varargin )

            %--------------------------------------------------------------
            % 1.) constructor of superclass
            %--------------------------------------------------------------
            exponents = zeros( 1, 7 );
            exponents( 1 ) = 2;
            objects@physical_values.physical_quantity_derived( exponents, varargin{ : } );

        end % function objects = squaremeter( varargin )

	end % methods

end % classdef squaremeter < physical_values.physical_quantity_derived
