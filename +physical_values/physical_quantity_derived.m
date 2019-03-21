%
% superclass for all derived physical quantities
%
% author: Martin F. Schiffner
% date: 2019-03-21
% modified: 2019-03-21
%
classdef physical_quantity_derived < physical_values.physical_value

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % methods
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	methods

        %------------------------------------------------------------------
        % constructor
        %------------------------------------------------------------------
        function objects = physical_quantity_derived( values )

            %--------------------------------------------------------------
            % 1.) constructor of superclass
            %--------------------------------------------------------------
            objects@physical_values.physical_value( values );

        end % function objects = physical_quantity_derived( values )

	end % methods

end % classdef physical_quantity_derived < physical_values.physical_value
