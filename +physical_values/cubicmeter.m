%
% superclass for all cubic meters
%
% author: Martin F. Schiffner
% date: 2019-03-27
% modified: 2019-03-27
%
classdef cubicmeter < physical_values.physical_quantity_derived

	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	% methods
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    methods

        %------------------------------------------------------------------
        % constructor
        %------------------------------------------------------------------
        function objects = cubicmeter( varargin )

            %--------------------------------------------------------------
            % 1.) constructor of superclass
            %--------------------------------------------------------------
            exponents = zeros( 1, 7 );
            exponents( 1 ) = 3;
            objects@physical_values.physical_quantity_derived( exponents, varargin{ : } );

        end % function objects = cubicmeter( varargin )

	end % methods

end % classdef cubicmeter < physical_values.physical_quantity_derived
