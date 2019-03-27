%
% superclass for all mass densities
%
% author: Martin F. Schiffner
% date: 2019-03-26
% modified: 2019-03-27
%
classdef mass_density < physical_values.physical_quantity_derived

	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	% methods
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    methods

        %------------------------------------------------------------------
        % constructor
        %------------------------------------------------------------------
        function objects = mass_density( varargin )

            %--------------------------------------------------------------
            % 1.) constructor of superclass
            %--------------------------------------------------------------
            exponents = zeros( 1, 8 );
            exponents( 1 ) = -3;
            exponents( 2 ) = 1;
            objects@physical_values.physical_quantity_derived( exponents, varargin{ : } );

        end % function objects = mass_density( varargin )

	end % methods

end % classdef mass_density < physical_values.physical_quantity_derived
