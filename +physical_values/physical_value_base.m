%
% superclass for all physical base quantities
%
% author: Martin F. Schiffner
% date: 2019-03-21
% modified: 2019-03-21
%
classdef physical_value_base < physical_values.physical_value

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % methods
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	methods

        %------------------------------------------------------------------
        % constructor
        %------------------------------------------------------------------
        function objects = physical_value_base( values )

            %--------------------------------------------------------------
            % 1.) constructor of superclass
            %--------------------------------------------------------------
            objects@physical_values.physical_value( values );

        end % function objects = physical_value_base( values )

	end % methods

end % classdef physical_value_base < physical_values.physical_value
