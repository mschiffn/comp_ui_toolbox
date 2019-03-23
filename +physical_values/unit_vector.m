%
% superclass for all unit vectors
%
% author: Martin F. Schiffner
% date: 2019-01-30
% modified: 2019-03-23
%
classdef unit_vector

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % properties
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	properties (SetAccess = private)

        % independent properties
        components ( 1, : ) double { mustBeReal, mustBeFinite, mustBeNonempty } = [0, 1]

    end % properties

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % methods
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	methods

        %------------------------------------------------------------------
        % constructor
        %------------------------------------------------------------------
        function objects = unit_vector( components )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % check number of arguments
            if nargin ~= 1
                return;
            end

            % ensure cell array for components
%             if ~iscell( components )
%                 components = { components };
%             end

            % ensure matrix argument
            if ~ismatrix( components ) || ~isreal( components ) || ~all( isfinite( components(:) ) )
                errorStruct.message     = 'components must be a real-valued finite matrix!';
                errorStruct.identifier	= 'unit_vector:NoRealFiniteMatrix';
                error( errorStruct );
            end

            % ensure l2-norms of unity
            norms = sqrt( sum( abs( components ).^2, 2 ) );
            if ~all( abs( norms - 1 ) < eps )
                errorStruct.message     = 'Rows of the argument must be unit vectors!';
                errorStruct.identifier	= 'unit_vector:NoRealMatrix';
                error( errorStruct );
            end

            %--------------------------------------------------------------
            % 2.) create unit vectors
            %--------------------------------------------------------------
            % construct column vector of objects
            N_objects = size( components, 1 );
            objects = repmat( objects, [ N_objects, 1 ] );

            % set independent properties
            for index_object = 1:N_objects
                objects( index_object ).components = components( index_object, : );
            end

        end % function objects = unit_vector( components )

        %------------------------------------------------------------------
        % cell array (overload cell function)
        %------------------------------------------------------------------
        function results = cell( unit_vectors )

            % create cell array of equal size
            results = cell( size( unit_vectors ) );

            % extract values
            for index_object = 1:numel( unit_vectors )
                results{ index_object } = unit_vectors( index_object ).components;
            end

        end % function results = cell( unit_vectors )

        %------------------------------------------------------------------
        % double-precision arrays (overload double function)
        %------------------------------------------------------------------
        function results = double( unit_vectors )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % ensure identical number of dimensions
            N_dimensions = cellfun( @(x) numel(x), cell( unit_vectors ) );
            if ~all( N_dimensions( : ) == N_dimensions( 1 ) )
                errorStruct.message     = 'One argument must be numeric and one argument must be physical_values.physical_value!';
                errorStruct.identifier	= 'double:Arguments';
                error( errorStruct );
            end

            %--------------------------------------------------------------
            % 2.) return double-precision array
            %--------------------------------------------------------------
            % initialize results
            % TODO: fix
            results = zeros( [ size( unit_vectors ), N_dimensions( 1 ) ] );

            % extract values
            for index_object = 1:numel( unit_vectors )
                results( index_object ) = unit_vectors( index_object ).value;
            end

        end % function results = double( unit_vectors )

        %------------------------------------------------------------------
        % element-wise multiplication (overload times function)
        %------------------------------------------------------------------
        function results = times( inputs_1, inputs_2 )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            if isa( inputs_1, 'physical_values.unit_vector' ) && ( isnumeric( inputs_2 ) || isa( inputs_2, 'physical_values.physical_value' ) )
                unit_vectors = inputs_1;
                numbers_in = inputs_2;
            elseif ( isnumeric( inputs_1 ) || isa( inputs_1, 'physical_values.physical_value' ) ) && isa( inputs_2, 'physical_values.unit_vector' )
                unit_vectors = inputs_2;
                numbers_in = inputs_1;
            else
                errorStruct.message     = 'One argument must be numeric or physical_values.physical_value and one argument must be physical_values.unit_vector!';
                errorStruct.identifier	= 'times:Arguments';
                error( errorStruct );
            end

            % multiple unit_vectors / single numbers_in
            if ~isscalar( unit_vectors ) && isscalar( numbers_in )
                numbers_in = repmat( numbers_in, size( unit_vectors ) );
            end

            % single unit_vectors / multiple numbers_in
            if isscalar( unit_vectors ) && ~isscalar( numbers_in )
                unit_vectors = repmat( unit_vectors, size( numbers_in ) );
            end

            % ensure equal number of dimensions and sizes
            auxiliary.mustBeEqualSize( unit_vectors, numbers_in );

            %--------------------------------------------------------------
            % 2.) compute results
            %--------------------------------------------------------------
            % create cell array
            size_unit_vectors = size( unit_vectors );
            results = cell( size_unit_vectors );

            % compute products
            for index_objects = 1:numel( unit_vectors )
                results{ index_objects } = unit_vectors( index_objects ).components * double( numbers_in( index_objects ) );
            end

            % return matrix for equal dimensions
            N_dimensions = cellfun( @(x) numel(x), results );
            if all( N_dimensions( : ) == N_dimensions( 1 ) )

                dim_singleton = find( size_unit_vectors == 1 );
                if isempty( dim_singleton )
                    size_result = [ size_unit_vectors, N_dimensions( 1 ) ];
                    dim_singleton = numel( size_unit_vectors ) + 1;
                else
                    size_result = size_unit_vectors;
                    size_result( dim_singleton ) = N_dimensions( 1 );
                end

                results_dbl = zeros( size_result );
                selector = repmat( { ':' }, [ 1, numel( size_result ) ] );
                for index_dim = 1:N_dimensions( 1 )
                    selector{ dim_singleton } = index_dim;
                    results_dbl( selector{ : } ) = cellfun( @(x) x( index_dim ), results );
                end

                results = results_dbl;
            end

        end % function objects_out = times( inputs_1, inputs_2 )

	end % methods

end % classdef unit_vector
