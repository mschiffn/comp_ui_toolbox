%
% superclass for all physical values
%
% author: Martin F. Schiffner
% date: 2019-01-16
% modified: 2019-02-13
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
            results = zeros( size( objects ) );

            % extract values
            for index_object = 1:numel( objects )
                results( index_object ) = objects( index_object ).value;
            end

        end % function results = double( objects )

        %------------------------------------------------------------------
        % unary plus (overload uplus function)
        %------------------------------------------------------------------
        function objects_out = uplus( objects_in )

            %--------------------------------------------------------------
            % 1.) perform uplus
            %--------------------------------------------------------------
            % create results of the same class
            objects_out = objects_in;

        end % function objects_out = uplus( objects_in )

        %------------------------------------------------------------------
        % unary minus (overload uminus function)
        %------------------------------------------------------------------
        function objects_out = uminus( objects_in )

            %--------------------------------------------------------------
            % 1.) perform negation
            %--------------------------------------------------------------
            % create results of the same class
            objects_out = objects_in;

            % invert signs
            for index_object = 1:numel( objects_in )
                objects_out( index_object ).value = - objects_in( index_object ).value;
            end

        end % function objects_out = uminus( objects_in )

        %------------------------------------------------------------------
        % absolute value (overload abs function)
        %------------------------------------------------------------------
        function results = abs( objects_in )

            % compute absolute values
            results = reshape( abs( [ objects_in.value ] ), size( objects_in ) );

        end % function objects_out = uminus( objects_in )

        %------------------------------------------------------------------
        % addition (overload plus function)
        %------------------------------------------------------------------
        function results = plus( objects_1, objects_2 )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % ensure equal classes
            if ~strcmp( class( objects_1 ), class( objects_2 ) )
                errorStruct.message     = 'Both arguments must have the same class!';
                errorStruct.identifier	= 'plus:ClassMismatch';
                error( errorStruct );
            end

            % ensure equal number of dimensions and sizes
            auxiliary.mustBeEqualSize( objects_1, objects_2 );
            % assertion: objects_1 and objects_2 have equal size

            %--------------------------------------------------------------
            % 2.) perform addition
            %--------------------------------------------------------------
            % create results of the same class
            results = objects_1;

            % add the physical values
            for index_object = 1:numel( objects_1 )
                results( index_object ).value = objects_1( index_object ).value + objects_2( index_object ).value;
            end

        end % function results = plus( objects_1, objects_2 )

        %------------------------------------------------------------------
        % subtraction (overload minus function)
        %------------------------------------------------------------------
        function results = minus( objects_1, objects_2 )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % ensure equal classes
            if ~strcmp( class( objects_1 ), class( objects_2 ) )
                errorStruct.message     = 'Both arguments must have the same class!';
                errorStruct.identifier	= 'minus:ClassMismatch';
                error( errorStruct );
            end

            % ensure equal number of dimensions and sizes
            auxiliary.mustBeEqualSize( objects_1, objects_2 );
            % assertion: objects_1 and objects_2 have equal size

            %--------------------------------------------------------------
            % 2.) perform subtraction
            %--------------------------------------------------------------
            % create results of the same class
            results = objects_1;

            % subtract the physical values
            for index_object = 1:numel( objects_1 )
                results( index_object ).value = objects_1( index_object ).value - objects_2( index_object ).value;
            end

        end % function results = minus( objects_1, objects_2 )

        %------------------------------------------------------------------
        % element-wise multiplication (overload times function)
        %------------------------------------------------------------------
        function objects_out = times( inputs_1, inputs_2 )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            if isa( inputs_1, 'physical_values.physical_value' ) && isnumeric( inputs_2 )
                objects_in = inputs_1;
                numbers_in = inputs_2;
            elseif  isnumeric( inputs_1 ) && isa( inputs_2, 'physical_values.physical_value' )
                objects_in = inputs_2;
                numbers_in = inputs_1;
            else
                errorStruct.message     = 'One argument must be numeric and one argument must be physical_values.physical_value!';
                errorStruct.identifier	= 'times:Arguments';
                error( errorStruct );
            end

            % multiple objects times scalar
            if ~isscalar( objects_in ) && isscalar( numbers_in )
                numbers_in = repmat( numbers_in, size( objects_in ) );
            end

            % single object times multiple
            if isscalar( objects_in ) && ~isscalar( numbers_in )
                objects_in = repmat( objects_in, size( numbers_in ) );
            end

            % ensure equal number of dimensions and sizes
            auxiliary.mustBeEqualSize( objects_in, numbers_in );
            % assertion: objects_in and numbers_in have equal sizes

            %--------------------------------------------------------------
            % 2.) compute results
            %--------------------------------------------------------------
            % create physical values
            objects_out = objects_in;

            for index_objects = 1:numel( objects_in )
                objects_out( index_objects ).value = objects_in( index_objects ).value * numbers_in( index_objects );
            end

        end % function objects_out = times( inputs_1, inputs_2 )

        %------------------------------------------------------------------
        % right array division (overload rdivide function)
        %------------------------------------------------------------------
        function results = rdivide( numerators, denominators )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % ensure class physical_values.physical_value
            if isnumeric( numerators ) && isa( denominators, 'physical_values.physical_value' )
                errorStruct.message     = 'Denominators must not be physical_values.physical_value!';
                errorStruct.identifier	= 'rdivide:Arguments';
                error( errorStruct );
            end
            % assertion: numerators is not numeric or denominators is not physical_values.physical_value

            % multiple objects in numerator / single object in denominator
            if ~isscalar( numerators ) && isscalar( denominators )
                denominators = repmat( denominators, size( numerators ) );
            end

            % single object in numerator / multiple objects in denominator
            if isscalar( numerators ) && ~isscalar( denominators )
                numerators = repmat( numerators, size( denominators ) );
            end

            % ensure equal number of dimensions and sizes
            auxiliary.mustBeEqualSize( numerators, denominators );
            % assertion: numerators and denominators have equal sizes

            %--------------------------------------------------------------
            % 2.) compute results
            %--------------------------------------------------------------
            if isa( numerators, 'physical_values.physical_value' ) && isnumeric( denominators )

                % create physical values
                results = numerators;

                for index_objects = 1:numel( results )
                    results( index_objects ).value = numerators( index_objects ).value / denominators( index_objects );
                end
            else

                % initialize results with zeros
                results = zeros( size( numerators ) );

                for index_objects = 1:numel( results )
                    results( index_objects ) = numerators( index_objects ).value / denominators( index_objects ).value;
                end
            end
        end % function results = rdivide( numerators, denominators )

        %------------------------------------------------------------------
        % round toward positive infinity (overload ceil function)
        %------------------------------------------------------------------
        function objects_out = ceil( objects_in )

            % copy physical values
            objects_out = objects_in;

            for index_objects = 1:numel( objects_out )
                objects_out( index_objects ).value = ceil( objects_in( index_objects ).value );
            end
        end % function objects_out = ceil( objects_in )

        %------------------------------------------------------------------
        % round toward negative infinity (overload floor function)
        %------------------------------------------------------------------
        function objects_out = floor( objects_in )

            % copy physical values
            objects_out = objects_in;

            for index_objects = 1:numel( objects_out )
                objects_out( index_objects ).value = floor( objects_in( index_objects ).value );
            end
        end % function objects_out = floor( objects_in )

        %------------------------------------------------------------------
        % minimum (overload min function)
        %------------------------------------------------------------------
        function results = min( objects )

            % create results of the same class
            results = objects( 1 );

            % return maximum value
            results.value = min( [ objects.value ] );
        end

        %------------------------------------------------------------------
        % maximum (overload max function)
        %------------------------------------------------------------------
        function results = max( objects )

            % create results of the same class
            results = objects( 1 );

            % return maximum value
            results.value = max( [ objects.value ] );
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
        end % function objects = quantize( objects, delta )

        %------------------------------------------------------------------
        % unique values in array (overload unique function)
        %------------------------------------------------------------------
        function [ objects_out, ia, ic ] = unique( physical_values )

            [ C, ia, ic ] = unique( double( physical_values ) );

            objects_out = repmat( physical_values( 1 ), size( C ) );
            for index_object = 1:numel( objects_out )
                objects_out( index_object ).value = C( index_object );
            end

        end % function [ objects_out, ia, ic ] = unique( physical_values )

        %------------------------------------------------------------------
        % display value of variable (overload disp function)
        %------------------------------------------------------------------
%         function objects = disp( objects )
% 
%             % quantize values
%             N_objects = numel( objects );
%             for index_object = 1:N_objects
%                 disp( objects( index_object ).value );
%             end
%         end

	end % methods

end % classdef physical_value
