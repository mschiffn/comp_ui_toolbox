%
% superclass for all voltages
%
% author: Martin F. Schiffner
% date: 2019-01-15
% modified: 2019-05-16
%
classdef squarevolt < physical_values.physical_quantity_derived

	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	%% methods
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    methods

        %------------------------------------------------------------------
        % constructor
        %------------------------------------------------------------------
        function objects = squarevolt( varargin )

            %--------------------------------------------------------------
            % 1.) constructor of superclass
            %--------------------------------------------------------------
            exponents = zeros( 1, 8 );
            exponents( 1 ) = 4;     % length
            exponents( 2 ) = 2;     % mass
            exponents( 3 ) = -6;    % time
            exponents( 4 ) = -2;    % electric current
            objects@physical_values.physical_quantity_derived( exponents, varargin{ : } );

        end % function objects = squarevolt( varargin )

	end % methods

end % classdef squarevolt < physical_values.physical_quantity_derived
