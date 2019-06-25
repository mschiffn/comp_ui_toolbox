%
% superclass for all square seconds
%
% author: Martin F. Schiffner
% date: 2019-06-13
% modified: 2019-06-13
%
classdef squaresecond < physical_values.physical_quantity_derived

	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	% methods
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    methods

        %------------------------------------------------------------------
        % constructor
        %------------------------------------------------------------------
        function objects = squaresecond( varargin )

            %--------------------------------------------------------------
            % 1.) constructor of superclass
            %--------------------------------------------------------------
            exponents = zeros( 1, 8 );
            exponents( 3 ) = 2;    % time
            objects@physical_values.physical_quantity_derived( exponents, varargin{ : } );

        end % function objects = squaresecond( varargin )

	end % methods

end % classdef squaresecond < physical_values.physical_quantity_derived
