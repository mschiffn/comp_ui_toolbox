%
% superclass for all lengths
%
% author: Martin F. Schiffner
% date: 2019-02-11
% modified: 2019-03-21
%
classdef length < physical_values.physical_value_base

	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	% methods
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    methods

        %------------------------------------------------------------------
        % constructor
        %------------------------------------------------------------------
        function objects = length( values )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % check number of arguments
            if nargin ~= 1
                errorStruct.message     = 'The number of arguments must equal unity!';
                errorStruct.identifier	= 'length:Arguments';
                error( errorStruct );
            end

            %--------------------------------------------------------------
            % 2.) constructor of superclass
            %--------------------------------------------------------------
            objects@physical_values.physical_value_base( values );

        end

	end % methods

end % classdef length < physical_values.physical_value_base
