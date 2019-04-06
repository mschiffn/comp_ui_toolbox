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
% modified: 2019-04-04
%
classdef physical_quantity < physical_values.transparent_container

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% properties
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	properties (SetAccess = private)

        % independent properties
        exponents ( 1, 8 ) double

    end % properties

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% methods
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	methods

        %------------------------------------------------------------------
        %% 1.) original functions
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
        %% 2.) prefix conversion
        %------------------------------------------------------------------
        % compute size in binary system
        function physical_quantity = convert_binary( physical_quantity, base, exponent )
            % TODO: change class
            physical_quantity.values = physical_quantity.values / base^exponent;
        end

        % kilo
        function physical_quantity = kilo( physical_quantity )

            % return result of binary conversion
            physical_quantity.values = convert_binary( physical_quantity.values, 10, 3 );
            str_class = class( physical_quantity );
        end

        % mega
        function physical_quantity = mega( physical_quantity )
            str_class = class( physical_quantity );
            str_class_mega = [ 'mega', str_class ];
            physical_quantity = convert_binary( physical_quantity, 10, 6 );
        end

        % giga
        function size_gigabyte = giga( objects )
            % return result of binary conversion
            size_gigabyte = convert_binary( objects, 10, 9 );
        end

        % kibi
        function size_kibibyte = kibi( objects )
            % return result of binary conversion
            size_kibibyte = convert_binary( objects, 2, 10 );
        end

        % mebi
        function size_mebibyte = mebi( objects )
            % return result of binary conversion
            size_mebibyte = convert_binary( objects, 2, 20 );
        end

        % gibi
        function size_gibibyte = gibi( objects )
            % return result of binary conversion
            size_gibibyte = convert_binary( objects, 2, 30 );
        end

        %------------------------------------------------------------------
        %% 3.) overload built-in type conversion functions
        %------------------------------------------------------------------
        % double-precision arrays
        function results = double( physical_quantity )
            results = double( physical_quantity.values );
        end

        % TODO: logical, struct

        %------------------------------------------------------------------
        %% 4.) overload built-in concatenation functions
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

                % skip empty argument
                if isempty( varargin{ index_arg } )
                    continue;
                end

                % ensure class physical_values.physical_quantity
                if ~isa( varargin{ index_arg }, 'physical_values.physical_quantity' )
                    % perform concatenation
                    physical_quantity_ref.values = cat( dim, physical_quantity_ref.values, varargin{ index_arg } );
                else
                    % ensure equal physical unit
                    if ~isequal( exponents_ref, varargin{ index_arg }.exponents )
                        errorStruct.message = sprintf( 'varargin{ %d } has incompatible physical unit!', index_arg );
                        errorStruct.identifier = 'cat:Arguments';
                        error( errorStruct );
                    end
                    % perform concatenation
                    physical_quantity_ref.values = cat( dim, physical_quantity_ref.values, varargin{ index_arg }.values );
                end

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
        %% 5.)
        %------------------------------------------------------------------
        % redefine subscripted assignment
        function physical_quantity = subsasgn( physical_quantity, S, B )

            if isempty( physical_quantity )
                physical_quantity = B;
            else
                if isa( B, 'physical_values.physical_quantity' )
                    mustBeCompatible( physical_quantity, B );
                    physical_quantity.values = subsasgn( physical_quantity.values, S, B.values );
                else
                    physical_quantity.values = subsasgn( physical_quantity.values, S, double( B ) );
                end
            end

        end

        %------------------------------------------------------------------
        %% 6.) overload built-in property validation functions
        %------------------------------------------------------------------
        % validate that value is greater than another value or issue error
        function mustBeGreaterThan( physical_quantity_1, physical_quantity_2 )
            mustBeCompatible( physical_quantity_1, physical_quantity_2 );
            mustBeGreaterThan@physical_values.transparent_container( physical_quantity_1, physical_quantity_2 );
        end

        % validate that value is greater than or equal to another value or issue error
        function mustBeGreaterThanOrEqual( physical_quantity_1, physical_quantity_2 )
            mustBeCompatible( physical_quantity_1, physical_quantity_2 );
            mustBeGreaterThanOrEqual@physical_values.transparent_container( physical_quantity_1, physical_quantity_2 );
        end

        % validate that value is less than another value or issue error
        function mustBeLessThan( physical_quantity_1, physical_quantity_2 )
            mustBeCompatible( physical_quantity_1, physical_quantity_2 );
            mustBeLessThan@physical_values.transparent_container( physical_quantity_1, physical_quantity_2 );
        end

        % validate that value is less than or equal to another value or issue error
        function mustBeLessThanOrEqual( physical_quantity_1, physical_quantity_2 )
            mustBeCompatible( physical_quantity_1, physical_quantity_2 );
            mustBeLessThanOrEqual@physical_values.transparent_container( physical_quantity_1, physical_quantity_2 );
        end

        %------------------------------------------------------------------
        %% 7.) overload built-in functions that maintain the physical unit
        %------------------------------------------------------------------
        % complex conjugate
        function physical_quantity = conj( physical_quantity )
            physical_quantity.values = conj( physical_quantity.values );
        end

        % transpose vector or matrix
        function physical_quantity = transpose( physical_quantity )
            physical_quantity.values = transpose( physical_quantity.values );
        end

        % complex conjugate transpose
        function physical_quantity = ctranspose( physical_quantity )
            physical_quantity.values = ctranspose( physical_quantity.values );
        end

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

        % sum of array elements
        function physical_quantity = sum( physical_quantity, varargin )
            physical_quantity.values = sum( physical_quantity.values, varargin{ : } );
        end

        % fast Fourier transform
        function physical_quantity = fft( physical_quantity, varargin )
            physical_quantity.values = fft( physical_quantity.values, varargin{ : } );
        end

        % TODO: fft2, fftn, ifft, ifft2, ifftn

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

        % vector creation, array subscripting, and for-loop iteration
        function physical_quantity_start = colon( physical_quantity_start, varargin )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % specify step size and stop value
            if nargin <= 2
                step = physical_quantity_start( 1 );
                step.values = 1;
                physical_quantity_stop = varargin{ 1 };
            else
                step = varargin{ 1 };
                physical_quantity_stop = varargin{ 2 };
            end

            % ensure compatible physical units
            mustBeCompatible( physical_quantity_start, step, physical_quantity_stop );

            %--------------------------------------------------------------
            % 2.) perform colon operation
            %--------------------------------------------------------------
            physical_quantity_start.values = physical_quantity_start.values:step.values:physical_quantity_stop.values;

        end

        % display value of variable
%         function disp( physical_quantity )
%             str_class = class( physical_quantity );
%             temp = physical_quantity / eval( str_class );
%             disp( temp );
%         end

        %------------------------------------------------------------------
        %% 8.) overload functions that potentially change the physical unit
        %------------------------------------------------------------------
        % matrix determinant
        function physical_quantity = det( physical_quantity )
            physical_quantity.exponents = size( physical_quantity.values, 1 ) * physical_quantity.exponents;
            physical_quantity.values = det( physical_quantity.values );
            physical_quantity = determine_class( physical_quantity );
        end

        % element-wise power
        function physical_quantity = power( physical_quantity, power )

            % ensure numeric nonempty power
% TODO: scalar vs matrix
            if ~( isnumeric( power ) && isscalar( power ) ) || isempty( power)
                errorStruct.message     = 'power must be numeric and nonempty!';
                errorStruct.identifier	= 'power:InvalidPower';
                error( errorStruct );
            end

            % compute element-wise power
            physical_quantity.exponents = power * physical_quantity.exponents;
            physical_quantity.values = physical_quantity.values.^power;
            physical_quantity = determine_class( physical_quantity );

        end

        % square root
        function physical_quantity = sqrt( physical_quantity )
            physical_quantity.exponents = physical_quantity.exponents / 2;
            physical_quantity.values = sqrt( physical_quantity.values );
            physical_quantity = determine_class( physical_quantity );
        end

        % element-wise multiplication
        function arg_1 = times( arg_1, arg_2 )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            if isa( arg_1, 'physical_values.physical_quantity' ) && ~isa( arg_2, 'physical_values.physical_quantity' )
                arg_1.values = arg_1.values .* arg_2;
            elseif ~isa( arg_1, 'physical_values.physical_quantity' ) && isa( arg_2, 'physical_values.physical_quantity' )
                arg_2.values = arg_1 .* arg_2.values;
                arg_1 = arg_2;
            elseif isa( arg_1, 'physical_values.physical_quantity' ) && isa( arg_2, 'physical_values.physical_quantity' )
                arg_1.exponents = arg_1.exponents + arg_2.exponents;
                arg_1.values = arg_1.values .* arg_2.values;
                arg_1 = determine_class( arg_1 );
            end

        end

        % right array division
        function arg_1 = rdivide( arg_1, arg_2 )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            if isa( arg_1, 'physical_values.physical_quantity' ) && ~isa( arg_2, 'physical_values.physical_quantity' )
                arg_1.values = arg_1.values ./ arg_2;
            elseif ~isa( arg_1, 'physical_values.physical_quantity' ) && isa( arg_2, 'physical_values.physical_quantity' )
                arg_2.exponents = - arg_2.exponents;
                arg_2.values = arg_1 ./ arg_2.values;
                arg_1 = determine_class( arg_2 );
            elseif isa( arg_1, 'physical_values.physical_quantity' ) && isa( arg_2, 'physical_values.physical_quantity' )
                arg_1.exponents = arg_1.exponents - arg_2.exponents;
                arg_1.values = arg_1.values ./ arg_2.values;
                arg_1 = determine_class( arg_1 );
            end

        end

        % matrix multiplication
        function arg_1 = mtimes( arg_1, arg_2 )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            if isa( arg_1, 'physical_values.physical_quantity' ) && ~isa( arg_2, 'physical_values.physical_quantity' )
                arg_1.values = arg_1.values * arg_2;
            elseif ~isa( arg_1, 'physical_values.physical_quantity' ) && isa( arg_2, 'physical_values.physical_quantity' )
                arg_2.values = arg_1 * arg_2.values;
                arg_1 = arg_2;
            elseif isa( arg_1, 'physical_values.physical_quantity' ) && isa( arg_2, 'physical_values.physical_quantity' )
                arg_1.exponents = arg_1.exponents + arg_2.exponents;
                arg_1.values = arg_1.values * arg_2.values;
                arg_1 = determine_class( arg_1 );
            end

        end

        % solve systems of linear equations
        function arg_1 = mrdivide( arg_1, arg_2 )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            if isa( arg_1, 'physical_values.physical_quantity' ) && ~isa( arg_2, 'physical_values.physical_quantity' )
                arg_1.values = arg_1.values / arg_2;
            elseif ~isa( arg_1, 'physical_values.physical_quantity' ) && isa( arg_2, 'physical_values.physical_quantity' )
                arg_2.exponents = - arg_2.exponents;
                arg_2.values = arg_1 / arg_2.values;
                arg_1 = determine_class( arg_2 );
            elseif isa( arg_1, 'physical_values.physical_quantity' ) && isa( arg_2, 'physical_values.physical_quantity' )
                arg_1.exponents = arg_1.exponents - arg_2.exponents;
                arg_1.values = arg_1.values / arg_2.values;
                arg_1 = determine_class( arg_1 );
            end

        end

        %------------------------------------------------------------------
        %% 9.) overload logical functions
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
