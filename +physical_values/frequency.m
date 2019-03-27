%
% superclass for all time frequencies
%
% author: Martin F. Schiffner
% date: 2019-01-15
% modified: 2019-03-27
%
classdef frequency < physical_values.physical_quantity_derived

	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	% methods
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    methods

        %------------------------------------------------------------------
        % constructor
        %------------------------------------------------------------------
        function objects = frequency( varargin )

            %--------------------------------------------------------------
            % 1.) constructor of superclass
            %--------------------------------------------------------------
            exponents = zeros( 1, 8 );
            exponents( 3 ) = -1;
            objects@physical_values.physical_quantity_derived( exponents, varargin{ : } );

        end % function objects = frequency( varargin )

	end % methods

end % classdef frequency < physical_values.physical_quantity_derived
