%
% superclass for physical quantities with the unit unity per meter
%
% author: Martin F. Schiffner
% date: 2019-04-07
% modified: 2020-01-14
%
classdef unity_per_meter < physical_values.physical_quantity_derived

	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	% methods
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    methods

        %------------------------------------------------------------------
        % constructor
        %------------------------------------------------------------------
        function objects = unity_per_meter( varargin )

            %--------------------------------------------------------------
            % 1.) constructor of superclass
            %--------------------------------------------------------------
            exponents = zeros( 1, 8 );
            exponents( 1 ) = -1; % length
            objects@physical_values.physical_quantity_derived( exponents, varargin{ : } );

        end

	end % methods

end % classdef unity_per_meter < physical_values.physical_quantity_derived
