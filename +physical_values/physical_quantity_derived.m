%
% superclass for all derived physical quantities
%
% author: Martin F. Schiffner
% date: 2019-03-21
% modified: 2019-03-27
%
classdef physical_quantity_derived < physical_values.physical_quantity

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % methods
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	methods

        %------------------------------------------------------------------
        % constructor
        %------------------------------------------------------------------
        function objects = physical_quantity_derived( exponents, varargin )

            %--------------------------------------------------------------
            % 1.) constructor of superclass
            %--------------------------------------------------------------
            objects@physical_values.physical_quantity( exponents, varargin{ : } );

        end % function objects = physical_quantity_derived( exponents, varargin )

	end % methods

end % classdef physical_quantity_derived < physical_values.physical_quantity
