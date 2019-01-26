%
% superclass for all apodization weights
%
% author: Martin F. Schiffner
% date: 2019-01-14
% modified: 2019-01-22
%
classdef apodization_weight < physical_values.physical_value

	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	% methods
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    methods

        %------------------------------------------------------------------
        % constructor
        %------------------------------------------------------------------
        function obj = apodization_weight( values )

            % check number of arguments
            if nargin ~= 1
                errorStruct.message     = 'The number of arguments must equal unity!';
                errorStruct.identifier	= 'apodization_weight:Arguments';
                error( errorStruct );
            end

            % constructor of superclass
            obj@physical_values.physical_value( values );
        end
	end % methods

end % classdef apodization_weight
