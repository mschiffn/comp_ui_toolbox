%
% superclass for all velocities
%
% author: Martin F. Schiffner
% date: 2019-03-26
% modified: 2020-01-14
%
classdef velocity < physical_values.physical_quantity_derived

	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	% methods
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    methods

        %------------------------------------------------------------------
        % constructor
        %------------------------------------------------------------------
        function objects = velocity( varargin )

            %--------------------------------------------------------------
            % 1.) constructor of superclass
            %--------------------------------------------------------------
            exponents = zeros( 1, 8 );
            exponents( 1 ) = 1;     % length
            exponents( 3 ) = -1;    % time
            objects@physical_values.physical_quantity_derived( exponents, varargin{ : } );

        end

	end % methods

end % classdef velocity < physical_values.physical_quantity_derived
