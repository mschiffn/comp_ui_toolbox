%
% superclass for all physical values
%
% author: Martin F. Schiffner
% date: 2019-01-16
% modified: 2019-01-22
%
classdef physical_value

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % properties
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	properties (SetAccess = private)

        % independent properties
        value ( 1, 1 ) double { mustBeReal, mustBeFinite }	% physical value
    end % properties

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % methods
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    methods

        %------------------------------------------------------------------
        % constructor
        %------------------------------------------------------------------
        function objects = physical_value( values )

            % check number of arguments
            if nargin ~= 1
                return;
            end

            % prevent emptyness of the argument
            mustBeNonempty( values );

            % construct column vector of objects
            N_elements = numel( values );
            objects = repmat( objects, [ N_elements, 1 ] );

            % set independent properties
            for index_element = 1:N_elements
                objects( index_element ).value = values( index_element );
            end

            % reshape to dimensions of the argument
            objects = reshape( objects, size( values ) );
        end

        %------------------------------------------------------------------
        % return value (overload double function)
        %------------------------------------------------------------------
        function result = double( objects )

            N_objects = numel( objects );
            result = zeros( N_objects, 1 );

            for index_object = 1:N_objects
                result( index_object ) = objects( index_object ).value;
            end
        end

        %------------------------------------------------------------------
        % subtraction (overload minus function)
        %------------------------------------------------------------------
        function result = minus( obj_1, obj_2 )

            result = physical_values.physical_value( double( obj_1 ) - double( obj_2 ) );
        end

        %------------------------------------------------------------------
        % addition (overload plus function)
        %------------------------------------------------------------------
        function result = plus( obj_1, obj_2 )

            result = physical_values.physical_value( double( obj_1 ) + double( obj_2 ) );
        end

        %------------------------------------------------------------------
        % comparison (overload greater than function)
        %------------------------------------------------------------------
        function tf = gt( obj_1, obj_2 )

            tf = false;
            if obj_1.value > obj_2.value
                tf = true;
            end
        end

        %------------------------------------------------------------------
        % quantize (overload greater than function)
        %------------------------------------------------------------------
        function objects = quantize( objects, delta )

            % check arguments
            % TODO: mustBeScalar( delta )
            mustBePositive( delta );

            % quantize values
            N_objects = numel( objects );
            for index_object = 1:N_objects
                objects( index_object ).value = round( objects( index_object ).value / delta ) * delta;
            end
        end
	end % methods

end % classdef physical_value
