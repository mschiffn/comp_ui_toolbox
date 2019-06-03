%
% superclass for all mass densities
%
% author: Martin F. Schiffner
% date: 2019-03-26
% modified: 2019-06-02
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
            exponents( 1 ) = -3;    % length
            exponents( 2 ) = 1;     % mass
            objects@physical_values.physical_quantity_derived( exponents, varargin{ : } );

        end % function objects = mass_density( varargin )

	end % methods

end % classdef mass_density < physical_values.physical_quantity_derived
