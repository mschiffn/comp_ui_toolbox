%
% superclass for all intervals of physical values
%
% author: Martin F. Schiffner
% date: 2019-01-21
% modified: 2019-02-01
%
classdef interval

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % properties
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	properties (SetAccess = private)

        % independent properties
        bounds ( 1, 2 ) physical_values.physical_value	% two time bounds define the interval
    end % properties

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % methods
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    methods

        %------------------------------------------------------------------
        % constructor
        %------------------------------------------------------------------
        function obj = interval( bounds )

            % check number of arguments
            if nargin ~= 1
                errorStruct.message     = 'The number of arguments must equal unity!';
                errorStruct.identifier	= 'interval:Arguments';
                error( errorStruct );
            end

            % prevent emptyness of the argument
            mustBeNonempty( bounds );

            % TODO: check class of bounds!
            % construct column vector of objects
            N_intervals = size( bounds, 1 );
            obj = repmat( obj, [ N_intervals, 1 ] );

            % check and set independent properties
            for index_interval = 1:N_intervals

                if bounds( index_interval, 2 ) > bounds( index_interval, 1 )

                    obj( index_interval ).bounds = bounds( index_interval, : );
                else

                    errorStruct.message     = 'bounds must increase strictly monotonic!';
                    errorStruct.identifier	= 'interval:NoIncrease';
                    error( errorStruct );
                end
            end
        end

        %------------------------------------------------------------------
        % length (overload abs function)
        %------------------------------------------------------------------
        function lengths = abs( objects )

            % compute lengths
            lengths = [ objects.bounds ];
            lengths = reshape( lengths( 2:2:end ) - lengths( 1:2:end ), size( objects) );
        end

        %------------------------------------------------------------------
        % quantization (overload quantize function)
        %------------------------------------------------------------------
        function objects = quantize( objects, deltas )

            % ensure equal number of dimensions and sizes
            auxiliary.mustBeEqualSize( objects, deltas )
            % assertion: objects and deltas have equal size

            % create matrix of boundary indices
            N_objects = numel( objects );
            q_bounds = zeros( N_objects, 2 );

            % quantize intervals
            for index_object = 1:N_objects

                % compute boundary indices
                q_bounds( index_object, 1 ) = floor( objects( index_object ).bounds( 1 ).value / deltas( index_object ).value );
                q_bounds( index_object, 2 ) = ceil( objects( index_object ).bounds( 2 ).value / deltas( index_object ).value );

                % compute quantized bounds
                objects( index_object ).bounds( 1 ) = q_bounds( index_object, 1 ) * deltas( index_object ).value;
                objects( index_object ).bounds( 2 ) = q_bounds( index_object, 2 ) * deltas( index_object ).value;
            end
        end

	end % methods

end % classdef interval
