%
% superclass for all physical base quantities
%
% author: Martin F. Schiffner
% date: 2019-03-21
% modified: 2019-03-27
%
classdef physical_quantity_base < physical_values.physical_quantity

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % methods
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	methods

        %------------------------------------------------------------------
        % constructor
        %------------------------------------------------------------------
        function objects = physical_quantity_base( index_exponent, varargin )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % ensure physical base quantities
            exponents = zeros( 1, 7 );
            exponents( index_exponent ) = 1;

            %--------------------------------------------------------------
            % 2.) constructor of superclass
            %--------------------------------------------------------------
            objects@physical_values.physical_quantity( exponents, varargin{ : } );

        end % function objects = physical_quantity_base( index_exponent, varargin )

	end % methods

end % classdef physical_quantity_base < physical_values.physical_quantity
