%
% superclass for all intervals of physical values
%
% author: Martin F. Schiffner
% date: 2019-01-21
% modified: 2019-01-21
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

                    errorStruct.message     = 'bounds must be strictly monotonic increasing!';
                    errorStruct.identifier	= 'interval:NoIncrease';
                    error( errorStruct );
                end
            end
        end

        %------------------------------------------------------------------
        % length (overload abs function)
        %------------------------------------------------------------------
        function length = abs( obj )

            % create column vector of durations
            N_intervals = numel( obj );
            length = zeros( N_intervals, 1 );

            % compute durations
            for index_interval = 1:N_intervals
                length( index_interval ) = double( obj( index_interval ).bounds( 2 ) - obj( index_interval ).bounds( 1 ) );
            end

            % reshape to dimensions of the argument
            length = reshape( length, size( obj) );
        end

        %------------------------------------------------------------------
        % quantization (overload quantize function)
        %------------------------------------------------------------------
        function objects = quantize( objects, deltas )

            % ensure equal dimensions
            if numel( objects ) ~= numel( deltas )
                errorStruct.message     = 'The numbers of components in objects and deltas must match!';
                errorStruct.identifier	= 'quantize:DimensionMismatch';
                error( errorStruct );
            end

            % create matrix of boundary indices
            N_objects = numel( objects );
            q_bounds = zeros( N_objects, 2 );

            % quantize intervals
            for index_object = 1:N_objects

                % compute boundary indices
                q_bounds( index_object, 1 ) = floor( objects( index_object ).bounds( 1 ).value / deltas( index_object ) );
                q_bounds( index_object, 2 ) = ceil( objects( index_object ).bounds( 2 ).value / deltas( index_object ) );

                % compute quantized bounds
                objects( index_object ).bounds = physical_values.physical_value( q_bounds( index_object, : ) * deltas( index_object ) );
            end
        end
	end % methods

end % classdef interval
