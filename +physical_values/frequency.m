%
% superclass for all frequencies
%
% author: Martin F. Schiffner
% date: 2019-01-15
% modified: 2019-02-01
%
classdef frequency < physical_values.physical_value

	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	% methods
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    methods

        %------------------------------------------------------------------
        % constructor
        %------------------------------------------------------------------
        function obj = frequency( values )

            % check number of arguments
            if nargin ~= 1
                errorStruct.message     = 'The number of arguments must equal unity!';
                errorStruct.identifier	= 'frequency:Arguments';
                error( errorStruct );
            end

            % check argument
            mustBePositive( values );

            % constructor of superclass
            obj@physical_values.physical_value( values );
        end

        %------------------------------------------------------------------
        % division (overload rdivide function)
        %------------------------------------------------------------------
        function objects = rdivide( numerators, denominators )

            % check arguments
            if ~isa( denominators, 'physical_values.frequency' )
                errorStruct.message     = 'numerators must be double, denominators must be physical_values.frequency!';
                errorStruct.identifier	= 'rdivide:Arguments';
                error( errorStruct );
            end
            % TODO: check size and dimensions
            % create times
            objects = physical_values.time( zeros( size( denominators ) ) );

            for index_objects = 1:numel( denominators )
                objects( index_objects ) = physical_values.time( 1 / denominators( index_objects ).value );
            end

        end

	end % methods

end % classdef frequency
