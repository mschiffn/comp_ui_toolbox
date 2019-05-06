%
% superclass for all unit vectors
%
% author: Martin F. Schiffner
% date: 2019-01-30
% modified: 2019-04-22
%
classdef ( InferiorClasses = {?physical_values.physical_quantity,?physical_values.meter} ) unit_vector

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% properties
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	properties (SetAccess = private)

        % independent properties
        components ( 1, : ) double { mustBeReal, mustBeFinite, mustBeNonempty } = [1, 0]

    end % properties

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% methods
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

            % convert matrix to cell array
            if ismatrix( components ) && isnumeric( components )
                components = mat2cell( components, ones( size( components, 1 ), 1 ) );
            end

            % ensure cell array for components
            if ~iscell( components )
                components = { components };
            end

            %--------------------------------------------------------------
            % 2.) create unit vectors
            %--------------------------------------------------------------
            % construct column vector of objects
            objects = repmat( objects, size( components ) );

            % set independent properties
            for index_object = 1:numel( components )

                % ensure l2-norms of unity
                if abs( norm( components{ index_object } ) - 1 ) >= eps
                    errorStruct.message     = sprintf( 'components{ %d } is not a unit vector!', index_object );
                    errorStruct.identifier	= 'unit_vector:NoUnitVector';
                    error( errorStruct );
                end

                % assign components
                objects( index_object ).components = components{ index_object };

            end % for index_object = 1:numel( components )

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
                errorStruct.message     = 'One argument must be numeric and one argument must be physical_values.physical_quantity!';
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
            if isa( inputs_1, 'math.unit_vector' ) && ( isnumeric( inputs_2 ) || isa( inputs_2, 'physical_values.physical_quantity' ) )
                unit_vectors = inputs_1;
                num_or_phys_val_in = inputs_2;
            elseif ( isnumeric( inputs_1 ) || isa( inputs_1, 'physical_values.physical_quantity' ) ) && isa( inputs_2, 'math.unit_vector' )
                unit_vectors = inputs_2;
                num_or_phys_val_in = inputs_1;
            else
                errorStruct.message     = 'One argument must be numeric or physical_values.physical_quantity and one argument must be math.unit_vector!';
                errorStruct.identifier	= 'times:Arguments';
                error( errorStruct );
            end

            % multiple unit_vectors / single num_or_phys_val_in
            if ~isscalar( unit_vectors ) && isscalar( num_or_phys_val_in )
                num_or_phys_val_in = repmat( num_or_phys_val_in, size( unit_vectors ) );
            end

            % single unit_vectors / multiple num_or_phys_val_in
            if isscalar( unit_vectors ) && ~isscalar( num_or_phys_val_in )
                unit_vectors = repmat( unit_vectors, size( num_or_phys_val_in ) );
            end

            % ensure equal number of dimensions and sizes
            auxiliary.mustBeEqualSize( unit_vectors, num_or_phys_val_in );

            %--------------------------------------------------------------
            % 2.) compute products
            %--------------------------------------------------------------
            % create cell array
            size_unit_vectors = size( unit_vectors );
            results = cell( size_unit_vectors );

            % compute products
            for index_objects = 1:numel( unit_vectors )
                results{ index_objects } = unit_vectors( index_objects ).components .* num_or_phys_val_in( index_objects );
            end

            % return matrix instead of cell array for equal dimensions
            N_dimensions = cellfun( @(x) numel(x), results );
            if all( N_dimensions( : ) == N_dimensions( 1 ) )

                % convert cell array to matrix by vertical concatenation
                results = vertcat( results{ : } );

                dim_cat = find( size_unit_vectors == 1 );
                if isempty( dim_cat )
                    size_result = [ size_unit_vectors, N_dimensions( 1 ) ];
                    dim_cat = numel( size_unit_vectors ) + 1;
                else
                    size_result = size_unit_vectors;
                    size_result( dim_cat ) = N_dimensions( 1 );
                end

            end

        end % function results = times( inputs_1, inputs_2 )

	end % methods

end % classdef unit_vector
