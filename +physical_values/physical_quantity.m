%
% superclass for all physical quantities
%
% inspired by:
%  [1] physical-units-for-matlab
%      www.mathworks.com/matlabcentral/fileexchange/authors/101715
%      github.com/sky-s/physical-units-for-matlab
%
% author: Martin F. Schiffner
% date: 2019-01-16
% modified: 2019-03-27
%
classdef physical_quantity < physical_values.transparent_container

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % properties
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	properties (SetAccess = private)

        % independent properties
        exponents

    end % properties

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % methods
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	methods

        %------------------------------------------------------------------
        % 1.) original functions
        %------------------------------------------------------------------
        % constructor
        function object = physical_quantity( exponents, varargin )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % default exponents
            if nargin < 1
                exponents = zeros( 1, 8 );
                exponents( 1 ) = 1;
            end

            % default values
            if nargin < 2
                values = 1;
            else
                values = varargin{ 1 };
            end

            % ensure nonemptyness of the argument
            mustBeNonempty( values );
            mustBeReal( values );
            mustBeFinite( values );

            %--------------------------------------------------------------
            % 2.) constructor of superclass
            %--------------------------------------------------------------
            object@physical_values.transparent_container( values );

            %--------------------------------------------------------------
            % 3.) set independent properties
            %--------------------------------------------------------------
            object.exponents = exponents;

        end % function object = physical_quantity( exponents, varargin )

        % compatible
        function mustBeCompatible( physical_quantity_ref, varargin )

            % single argument is always compatible
            if nargin < 2
                errorStruct.message     = 'At least two arguments are required!';
                errorStruct.identifier	= 'mustBeCompatible:FewArguments';
                error( errorStruct );
            end

            % ensure class physical_values.physical_quantity
            if ~isa( physical_quantity_ref, 'physical_values.physical_quantity' )
                errorStruct.message = 'physical_quantity_ref must be physical_values.physical_quantity!';
                errorStruct.identifier = 'mustBeCompatible:NoPhysicalValue';
                error( errorStruct );
            end

            % use first argument as reference
            exponents_ref = physical_quantity_ref.exponents;

            % check exponents of remaining arguments
            for index_arg = 1:numel( varargin )

                % ensure class physical_values.physical_quantity
                if ~isa( varargin{ index_arg }, 'physical_values.physical_quantity' )
                    errorStruct.message = sprintf( 'varargin{ %d } is not physical_values.physical_quantity!', index_arg );
                    errorStruct.identifier = 'mustBeCompatible:NoPhysicalValue';
                    error( errorStruct );
                end

                % ensure equal number of dimensions and sizes of cell arrays
%                 auxiliary.mustBeEqualSize( physical_quantity_ref, varargin{ index_arg } );

                % ensure equal physical unit
                if ~isequal( exponents_ref, varargin{ index_arg }.exponents )
                    errorStruct.message = sprintf( 'varargin{ %d } has incompatible physical unit!', index_arg );
                    errorStruct.identifier = 'mustBeCompatible:Arguments';
                    error( errorStruct );
                end

            end % for index_arg = 1:numel( varargin )

        end % function mustBeCompatible( physical_quantity_ref, varargin )

        % determine class
        function result = determine_class( physical_quantity )

            %--------------------------------------------------------------
            % 1.) check if result is a physical quantity
            %--------------------------------------------------------------
            % TODO: tolerance
            % return double variable for zero exponents
            if all( abs( physical_quantity.exponents ) < eps )
                result = physical_quantity.values;
                return;
            end

            %--------------------------------------------------------------
            % 2.) assign class
            %--------------------------------------------------------------
            % investigate exponents and return appropriate classes
            if isequal( physical_quantity.exponents, [ 1, 0, 0, 0, 0, 0, 0, 0 ] )
                result = physical_values.meter( physical_quantity.values );
            elseif isequal( physical_quantity.exponents, [ 2, 0, 0, 0, 0, 0, 0, 0 ] )
                result = physical_values.squaremeter( physical_quantity.values );
            elseif isequal( physical_quantity.exponents, [ 3, 0, 0, 0, 0, 0, 0, 0 ] )
                result = physical_values.cubicmeter( physical_quantity.values );
            elseif isequal( physical_quantity.exponents, [ 0, 1, 0, 0, 0, 0, 0, 0 ] )
                result = physical_values.kilogram( physical_quantity.values );
            elseif isequal( physical_quantity.exponents, [ 0, 0, 1, 0, 0, 0, 0, 0 ] )
                result = physical_values.second( physical_quantity.values );
            elseif isequal( physical_quantity.exponents, [ 0, 0, -1, 0, 0, 0, 0, 0 ] )
                result = physical_values.hertz( physical_quantity.values );
            elseif isequal( physical_quantity.exponents, [ 0, 0, 0, 1, 0, 0, 0, 0 ] )
                result = physical_values.ampere( physical_quantity.values );
            elseif isequal( physical_quantity.exponents, [ 1, 0, -1, 0, 0, 0, 0, 0 ] )
                result = physical_values.meter_per_second( physical_quantity.values );
            elseif isequal( physical_quantity.exponents, [ 2, 1, -3, -1, 0, 0, 0, 0 ] )
                result = physical_values.volt( physical_quantity.values );
            else
                result = physical_values.physical_quantity_derived( physical_quantity.exponents, physical_quantity.values );
                warning( 'Physical unit of unknown class!' );
            end

        end

        %------------------------------------------------------------------
        % 2.) overload built-in type conversion functions
        %------------------------------------------------------------------
        function results = double( physical_quantity )
            results = double( physical_quantity.values );
        end

        %------------------------------------------------------------------
        % 3.) overload built-in concatenation functions
        %------------------------------------------------------------------
        % concatenate arrays along specified dimension
        function physical_quantity_ref = cat( dim, physical_quantity_ref, varargin )

            % ensure class physical_values.physical_quantity
            if ~isa( physical_quantity_ref, 'physical_values.physical_quantity' )
                errorStruct.message = 'physical_quantity_ref must be physical_values.physical_quantity!';
                errorStruct.identifier = 'cat:NoPhysicalValue';
                error( errorStruct );
            end

            % use first argument as reference
            exponents_ref = physical_quantity_ref.exponents;
        
            % check exponents of remaining arguments
            for index_arg = 1:numel( varargin )

                % ensure class physical_values.physical_quantity
                if ~isa( varargin{ index_arg }, 'physical_values.physical_quantity' )
                    errorStruct.message = sprintf( 'varargin{ %d } is not physical_values.physical_quantity!', index_arg );
                    errorStruct.identifier = 'cat:NoPhysicalValue';
                    error( errorStruct );
                end

                % ensure equal physical unit
                if ~isequal( exponents_ref, varargin{ index_arg }.exponents )
                    errorStruct.message = sprintf( 'varargin{ %d } has incompatible physical unit!', index_arg );
                    errorStruct.identifier = 'cat:Arguments';
                    error( errorStruct );
                end

                % perform concatenation
                physical_quantity_ref.values = cat( dim, physical_quantity_ref.values, varargin{ index_arg }.values );

            end % for index_arg = 1:numel( varargin )

        end

        % concatenate arrays horizontally
        function physical_quantity = horzcat( varargin )
            physical_quantity = cat( 2, varargin{ : } );
        end

        % concatenate arrays vertically
        function physical_quantity = vertcat( varargin )
            physical_quantity = cat( 1, varargin{ : } );
        end

        %------------------------------------------------------------------
        % 4.) overload built-in functions that maintain the physical unit
        %------------------------------------------------------------------
        % unary plus
        function physical_quantity = uplus( physical_quantity )
        end

        % unary minus
        function physical_quantity = uminus( physical_quantity )
            physical_quantity.values = - physical_quantity.values;
        end

        % absolute value
        function physical_quantity = abs( physical_quantity )
            physical_quantity.values = abs( physical_quantity.values );
        end

        % round toward positive infinity
        function physical_quantity = ceil( physical_quantity )
            physical_quantity.values = ceil( physical_quantity.values );
        end

        % round toward negative infinity
        function physical_quantity = floor( physical_quantity )
            physical_quantity.values = floor( physical_quantity.values );
        end

        % minimum elements of an array
        function varargout = min( physical_quantity, varargin )

            % TODO: exclude problematic cases
            varargout{1} = physical_quantity;
            [ varargout{1}.values, varargout{2:nargout} ] = min( physical_quantity.values, varargin{ : } );

        end

        % maximum elements of an array
        function varargout = max( physical_quantity, varargin )

            % TODO: exclude problematic cases
            varargout{1} = physical_quantity;
            [ varargout{1}.values, varargout{2:nargout} ] = max( physical_quantity.values, varargin{ : } );

        end

        % unique values in array
        function varargout = unique( physical_quantity, varargin )

            % extract unique physical quantities
            varargout{ 1 } = physical_quantity;
            [ varargout{ 1 }.values, varargout{ 2:nargout } ] = unique( physical_quantity.values, varargin{ : } );

        end

        % addition
        function physical_quantity_1 = plus( physical_quantity_1, physical_quantity_2 )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % ensure compatible physical units
            mustBeCompatible( physical_quantity_1, physical_quantity_2 );

            %--------------------------------------------------------------
            % 2.) perform addition
            %--------------------------------------------------------------
            physical_quantity_1.values = physical_quantity_1.values + physical_quantity_2.values;

        end

        % subtraction
        function physical_quantity_1 = minus( physical_quantity_1, physical_quantity_2 )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % ensure compatible physical units
            mustBeCompatible( physical_quantity_1, physical_quantity_2 );

            %--------------------------------------------------------------
            % 2.) perform subtraction
            %--------------------------------------------------------------
            physical_quantity_1.values = physical_quantity_1.values - physical_quantity_2.values;

        end

        % quantize
%         function objects = quantize( physical_quantity, delta )
% 
%             % check arguments
%             if ~isscalar( delta )
%                 errorStruct.message     = 'delta must be a scalar!';
%                 errorStruct.identifier	= 'quantize:NoScalar';
%                 error( errorStruct );
%             end
%             mustBePositive( delta );
% 
%             % quantize values
%             N_objects = numel( objects );
%             for index_object = 1:N_objects
%                 objects( index_object ).value = round( objects( index_object ).value / delta ) * delta;
%             end
%         end % function objects = quantize( objects, delta )

        % display value of variable
%         function disp( physical_quantity )
%             str_class = class( physical_quantity );
%             temp = physical_quantity / eval( str_class );
%             disp( temp );
%         end

        %------------------------------------------------------------------
        % 5.) overload functions that potentially change the physical unit
        %------------------------------------------------------------------
        % matrix determinant
        function physical_quantity = det( physical_quantity )

            physical_quantity.exponents = size( physical_quantity.values, 1 ) * physical_quantity.exponents;
            physical_quantity.values = det( physical_quantity.values );
            physical_quantity = determine_class( physical_quantity );
        end

        % element-wise multiplication
        function arg_1 = times( arg_1, arg_2 )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            if isa( arg_1, 'physical_values.physical_quantity' ) && isnumeric( arg_2 )
                arg_1.values = arg_1.values .* arg_2;
            elseif isnumeric( arg_1 ) && isa( arg_2, 'physical_values.physical_quantity' )
                arg_2.values = arg_1 .* arg_2.values;
                arg_1 = arg_2;
            elseif isa( arg_1, 'physical_values.physical_quantity' ) && isa( arg_2, 'physical_values.physical_quantity' )
                arg_1.values = arg_1.values .* arg_2.values;
                arg_1.exponents = arg_1.exponents + arg_2.exponents;
                arg_1 = determine_class( arg_1 );
            else
                errorStruct.message     = 'One argument must be numeric and one argument must be physical_values.physical_quantity or both arguments must be physical_values.physical_quantity!';
                errorStruct.identifier	= 'mtimes:Arguments';
                error( errorStruct );
            end

        end

        % right array division
        function arg_1 = rdivide( arg_1, arg_2 )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            if isa( arg_1, 'physical_values.physical_quantity' ) && isnumeric( arg_2 )
                arg_1.values = arg_1.values ./ arg_2;
            elseif isnumeric( arg_1 ) && isa( arg_2, 'physical_values.physical_quantity' )
                arg_2.values = arg_1 ./ arg_2.values;
                arg_2.exponents = - arg_2.exponents;
                arg_1 = determine_class( arg_2 );
            elseif isa( arg_1, 'physical_values.physical_quantity' ) && isa( arg_2, 'physical_values.physical_quantity' )
                arg_1.values = arg_1.values ./ arg_2.values;
                arg_1.exponents = arg_1.exponents - arg_2.exponents;
                arg_1 = determine_class( arg_1 );
            else
                errorStruct.message     = 'One argument must be numeric and one argument must be physical_values.physical_quantity or both arguments must be physical_values.physical_quantity!';
                errorStruct.identifier	= 'mrdivide:Arguments';
                error( errorStruct );
            end

        end

        % matrix multiplication
        function arg_1 = mtimes( arg_1, arg_2 )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            if isa( arg_1, 'physical_values.physical_quantity' ) && isnumeric( arg_2 )
                arg_1.values = arg_1.values * arg_2;
            elseif isnumeric( arg_1 ) && isa( arg_2, 'physical_values.physical_quantity' )
                arg_2.values = arg_1 * arg_2.values;
                arg_1 = arg_2;
            elseif isa( arg_1, 'physical_values.physical_quantity' ) && isa( arg_2, 'physical_values.physical_quantity' )
                arg_1.values = arg_1.values * arg_2.values;
                arg_1.exponents = arg_1.exponents + arg_2.exponents;
                arg_1 = determine_class( arg_1 );
            else
                errorStruct.message     = 'One argument must be numeric and one argument must be physical_values.physical_quantity or both arguments must be physical_values.physical_quantity!';
                errorStruct.identifier	= 'mtimes:Arguments';
                error( errorStruct );
            end

        end

        % solve systems of linear equations
        function arg_1 = mrdivide( arg_1, arg_2 )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            if isa( arg_1, 'physical_values.physical_quantity' ) && isnumeric( arg_2 )
                arg_1.values = arg_1.values / arg_2;
            elseif isnumeric( arg_1 ) && isa( arg_2, 'physical_values.physical_quantity' )
                arg_2.values = arg_1 / arg_2.values;
                arg_2.exponents = - arg_2.exponents;
                arg_1 = determine_class( arg_2 );
            elseif isa( arg_1, 'physical_values.physical_quantity' ) && isa( arg_2, 'physical_values.physical_quantity' )
                arg_1.values = arg_1.values / arg_2.values;
                arg_1.exponents = arg_1.exponents - arg_2.exponents;
                arg_1 = determine_class( arg_1 );
            else
                errorStruct.message     = 'One argument must be numeric and one argument must be physical_values.physical_quantity or both arguments must be physical_values.physical_quantity!';
                errorStruct.identifier	= 'mrdivide:Arguments';
                error( errorStruct );
            end

        end

        %------------------------------------------------------------------
        % 6.) overload logical functions
        %------------------------------------------------------------------
        % determine greater than
        function results = gt( physical_quantity_1, physical_quantity_2 )

            % ensure compatible physical units
            mustBeCompatible( physical_quantity_1, physical_quantity_2 );

            % perform check
            results = physical_quantity_1.values > physical_quantity_2.values;
        end

        % determine equality
        function result = eq( physical_quantity_1, physical_quantity_2 )

            % ensure compatible physical units
            mustBeCompatible( physical_quantity_1, physical_quantity_2 );

            % perform check
            result = physical_quantity_1.values == physical_quantity_2.values;
        end

        % determine greater than or equal to
        function result = ge( physical_quantity_1, physical_quantity_2 )

            % ensure compatible physical units
            mustBeCompatible( physical_quantity_1, physical_quantity_2 );

            % perform check
            result = physical_quantity_1.values >= physical_quantity_2.values;
        end

        % determine less than or equal to
        function result = le( physical_quantity_1, physical_quantity_2 )

            % ensure compatible physical units
            mustBeCompatible( physical_quantity_1, physical_quantity_2 );

            % perform check
            result = physical_quantity_1.values <= physical_quantity_2.values;
        end

        % determine less than
        function result = lt( physical_quantity_1, physical_quantity_2 )

            % ensure compatible physical units
            mustBeCompatible( physical_quantity_1, physical_quantity_2 );

            % perform check
            result = physical_quantity_1.values < physical_quantity_2.values;
        end

        % determine inequality
        function result = ne( physical_quantity_1, physical_quantity_2 )

            % ensure compatible physical units
            mustBeCompatible( physical_quantity_1, physical_quantity_2 );

            % perform check
            result = physical_quantity_1.values ~= physical_quantity_2.values;
        end

        % find logical NOT
        function result = not( physical_quantity )
            result = ~physical_quantity.values;
        end

	end % methods

end % classdef physical_quantity
