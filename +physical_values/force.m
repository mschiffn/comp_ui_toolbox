%
% superclass for all time frequencies
%
% author: Martin F. Schiffner
% date: 2019-05-04
% modified: 2019-05-04
%
classdef force < physical_values.physical_quantity_derived

	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	%% methods
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    methods

        %------------------------------------------------------------------
        % constructor
        %------------------------------------------------------------------
        function objects = force( varargin )

            %--------------------------------------------------------------
            % 1.) constructor of superclass
            %--------------------------------------------------------------
            exponents = zeros( 1, 8 );
            exponents( 1 ) = 1;     % length
            exponents( 2 ) = 1;     % mass
            exponents( 3 ) = -2;	% time
            objects@physical_values.physical_quantity_derived( exponents, varargin{ : } );

        end % function objects = force( varargin )

	end % methods

end % classdef force < physical_values.physical_quantity_derived
