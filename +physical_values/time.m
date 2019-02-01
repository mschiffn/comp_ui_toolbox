%
% superclass for all times
%
% author: Martin F. Schiffner
% date: 2019-01-14
% modified: 2019-02-01
%
classdef time < physical_values.physical_value

	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	% methods
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    methods

        %------------------------------------------------------------------
        % constructor
        %------------------------------------------------------------------
        function obj = time( values )

            % check number of arguments
            if nargin ~= 1
                errorStruct.message     = 'The number of arguments must equal unity!';
                errorStruct.identifier	= 'time:Arguments';
                error( errorStruct );
            end

            % check argument
            mustBeNonnegative( values );

            % constructor of superclass
            obj@physical_values.physical_value( values );
        end

        %------------------------------------------------------------------
        % division (overload rdivide function)
        %------------------------------------------------------------------
        function objects = rdivide( numerators, denominators )

            % check arguments
            if ~isa( denominators, 'physical_values.time' )
                errorStruct.message     = 'numerators must be double, denominators must be physical_values.time!';
                errorStruct.identifier	= 'rdivide:Arguments';
                error( errorStruct );
            end
            % TODO: check size and dimensions
            denominators = repmat( denominators, size( numerators ) );

            % create frequencies
            objects = physical_values.frequency( ones( size( denominators ) ) );

            for index_objects = 1:numel( denominators )
                objects( index_objects ) = physical_values.frequency( numerators( index_objects ) / denominators( index_objects ).value );
            end

        end

	end % methods

end % classdef time
