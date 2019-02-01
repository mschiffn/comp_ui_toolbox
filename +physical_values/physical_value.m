%
% superclass for all physical values
%
% author: Martin F. Schiffner
% date: 2019-01-16
% modified: 2019-02-01
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

            % reshape to size of the argument
            objects = reshape( objects, size( values ) );
        end

        %------------------------------------------------------------------
        % return value (overload double function)
        %------------------------------------------------------------------
        function results = double( objects )

            % create results of equal size
            N_objects = numel( objects );
            results = zeros( size( objects ) );

            % extract values
            for index_object = 1:N_objects
                results( index_object ) = objects( index_object ).value;
            end
        end

        %------------------------------------------------------------------
        % subtraction (overload minus function)
        %------------------------------------------------------------------
        function results = minus( objects_1, objects_2 )

            % ensure equal number of dimensions and sizes
            auxiliary.mustBeEqualSize( objects_1, objects_2 )
            % assertion: objects_1 and objects_2 have equal size

            % create results of the same class
            results = objects_1;

            % subtract the physical values
            for index_object = 1:numel( objects_1 )
                results( index_object ).value = objects_1( index_object ).value - objects_2( index_object ).value;
            end
        end

        %------------------------------------------------------------------
        % addition (overload plus function)
        %------------------------------------------------------------------
        function results = plus( objects_1, objects_2 )

            % ensure equal number of dimensions and sizes
            auxiliary.mustBeEqualSize( objects_1, objects_2 )
            % assertion: objects_1 and objects_2 have equal size

            % create results of the same class
            results = objects_1;

            % subtract the physical values
            for index_object = 1:numel( objects_1 )
                results( index_object ).value = objects_1( index_object ).value + objects_2( index_object ).value;
            end
        end

        %------------------------------------------------------------------
        % comparison (overload greater than function)
        %------------------------------------------------------------------
        function results = gt( objects_1, objects_2 )

            % ensure equal number of dimensions and sizes
            auxiliary.mustBeEqualSize( objects_1, objects_2 )
            % assertion: objects_1 and objects_2 have equal size

            % initialize results
            results = false( size( objects_1 ) );

            % compare the physical values
            for index_object = 1:numel( objects_1 )
                if objects_1( index_object ).value > objects_2( index_object ).value
                    results( index_object ) = true;
                end
            end
        end

        %------------------------------------------------------------------
        % quantize (overload quantize function)
        %------------------------------------------------------------------
        function objects = quantize( objects, delta )

            % check arguments
            if ~isscalar( delta )
                errorStruct.message     = 'delta must be a scalar!';
                errorStruct.identifier	= 'quantize:NoScalar';
                error( errorStruct );
            end
            mustBePositive( delta );

            % quantize values
            N_objects = numel( objects );
            for index_object = 1:N_objects
                objects( index_object ).value = round( objects( index_object ).value / delta ) * delta;
            end
        end

	end % methods

end % classdef physical_value
