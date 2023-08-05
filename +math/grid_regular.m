%
% superclass for all regular grids
%
% wikipedia: A regular grid is a tessellation of n-dimensional Euclidean space by congruent parallelotopes (e.g. bricks).
%
% author: Martin F. Schiffner
% date: 2018-01-23
% modified: 2023-08-05
%
classdef grid_regular < math.grid

	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	%% properties
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	properties (SetAccess = private)

        % independent properties
        offset_axis ( 1, : ) physical_values.length     % arbitrary offset
        cell_ref ( 1, 1 ) math.parallelotope            % elementary cell
        N_points_axis ( 1, : ) double { mustBeInteger, mustBePositive, mustBeNonempty } = [ 128, 128 ]	 % numbers of grid points along each coordinate axis (1)

    end % properties

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% methods
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	methods

        %------------------------------------------------------------------
        % constructor
        %------------------------------------------------------------------
        function objects = grid_regular( offset_axis, cells_ref, N_points_axis )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % ensure cell array for offset_axis
            if ~iscell( offset_axis )
                offset_axis = { offset_axis };
            end

            % ensure class math.parallelotope
            if ~isa( cells_ref, 'math.parallelotope' )
                errorStruct.message = 'cells_ref must be math.parallelotope!';
                errorStruct.identifier = 'grid_regular:NoParallelotopes';
                error( errorStruct );
            end

            % ensure cell array for N_points_axis
            if ~iscell( N_points_axis )
                N_points_axis = { N_points_axis };
            end

            % ensure equal number of dimensions and sizes
            [ offset_axis, cells_ref, N_points_axis ] = auxiliary.ensureEqualSize( offset_axis, cells_ref, N_points_axis );

            %--------------------------------------------------------------
            % 2.) compute positions of the grid points
            %--------------------------------------------------------------
            % specify cell array for positions
            positions = cell( size( offset_axis ) );

            % iterate regular grids
            for index_object = 1:numel( positions )

                % ensure nonempty positive integers
                mustBePositive( N_points_axis{ index_object } );
                mustBeInteger( N_points_axis{ index_object } );

                N_dimensions = size( offset_axis{ index_object }, 2 );
                N_points = prod( N_points_axis{ index_object }, 2 );

                positions{ index_object } = physical_values.meter( zeros( N_points, N_dimensions ) );

            end % for index_object = 1:numel( positions )

            %--------------------------------------------------------------
            % 3.) constructor of superclass
            %--------------------------------------------------------------
            objects@math.grid( positions );

            %--------------------------------------------------------------
            % 4.) set independent properties
            %--------------------------------------------------------------
            for index_object = 1:numel( objects )

                % ensure class physical_values.length
                if ~isa( offset_axis{ index_object }, 'physical_values.length' )
                    errorStruct.message     = 'offset_axis must be physical_values.length!';
                    errorStruct.identifier	= 'grid_regular:NoLength';
                    error( errorStruct );
                end

                % ensure equal number of dimensions and sizes
                auxiliary.mustBeEqualSize( offset_axis{ index_object }, cells_ref( index_object ).edge_lengths, N_points_axis{ index_object } );

                % set independent properties
                objects( index_object ).offset_axis = offset_axis{ index_object };
                objects( index_object ).cell_ref = cells_ref( index_object );
                objects( index_object ).N_points_axis = N_points_axis{ index_object };

                % set dependent properties
                objects( index_object ).positions = compute_positions( objects( index_object ) );

            end % for index_object = 1:numel( offset_axis )

        end % function objects = grid_regular( offset_axis, cells_ref, N_points_axis )

        %------------------------------------------------------------------
        % compute discrete positions of the grid points
        %------------------------------------------------------------------
        function positions = compute_positions( grids_regular )

            % create cell array
            positions = cell( size( grids_regular ) );

            % iterate regular grids
            for index_object = 1:numel( grids_regular )

                % calculate indices along each coordinate axis
                indices_linear = ( 1:grids_regular( index_object ).N_points );
                indices_axis = inverse_index_transform( grids_regular( index_object ), indices_linear );

                % compute Cartesian coordinates of grid points
                positions_rel = ( indices_axis - 1 ) * ( grids_regular( index_object ).cell_ref.edge_lengths .* grids_regular( index_object ).cell_ref.basis );
                positions{ index_object } = grids_regular( index_object ).offset_axis + positions_rel;

            end % for index_object = 1:numel( grids_regular )

            % avoid cell array for single regular grid
            if isscalar( grids_regular )
                positions = positions{ 1 };
            end

        end % function positions = compute_positions( grids_regular )

        %------------------------------------------------------------------
        % forward index transform
        %------------------------------------------------------------------
        function indices_linear = forward_index_transform( grids_regular, indices_axis )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % ensure class math.grid_regular
            if ~isa( grids_regular, 'math.grid_regular' )
                errorStruct.message = 'grids_regular must be math.grid_regular!';
                errorStruct.identifier = 'forward_index_transform:NoRegularGrids';
                error( errorStruct );
            end

            % ensure cell array for indices_axis
            if ~iscell( indices_axis )
                indices_axis = { indices_axis };
            end

            % ensure equal number of dimensions and sizes
            [ grids_regular, indices_axis ] = auxiliary.ensureEqualSize( grids_regular, indices_axis );

            %--------------------------------------------------------------
            % 2.) convert array indices into linear indices
            %--------------------------------------------------------------
            % specify cell array for indices_axis
            indices_linear = cell( size( grids_regular ) );

            % iterate regular grids
            for index_object = 1:numel( grids_regular )

                % ensure numeric and real-valued matrix for indices_axis{ index_object }
                if ~( isnumeric( indices_axis{ index_object } ) && isreal( indices_axis{ index_object } ) && ismatrix( indices_axis{ index_object } ) )
                    errorStruct.message = sprintf( 'indices_axis{ %d } must be a numeric and real-valued matrix!', index_object );
                    errorStruct.identifier = 'forward_index_transform:NoNumericAndRealMatrix';
                    error( errorStruct );
                end

                % ensure correct dimensionality
                if size( indices_axis{ index_object }, 2 ) ~= grids_regular( index_object ).N_dimensions
                    errorStruct.message = sprintf( 'indices_axis{ %d } must have %d columns!', index_object, grids_regular( index_object ).N_dimensions );
                    errorStruct.identifier = 'forward_index_transform:InvalidDimensionality';
                    error( errorStruct );
                end

                % ensure valid integers for indices_axis{ index_object }
                if ~( all( indices_axis{ index_object } == floor( indices_axis{ index_object } ), 'all' ) && all( indices_axis{ index_object } > 0, 'all' ) && all( indices_axis{ index_object } <= grids_regular( index_object ).N_points_axis, 'all' ) )
                    errorStruct.message = sprintf( 'indices_axis{ %d } must contain integers in the correct range!', index_object );
                    errorStruct.identifier = 'forward_index_transform:NoValidIntegers';
                    error( errorStruct );
                end

                temp = mat2cell( indices_axis{ index_object }, size( indices_axis{ index_object }, 1 ), ones( 1, grids_regular( index_object ).N_dimensions ) );
                indices_linear{ index_object } = sub2ind( grids_regular( index_object ).N_points_axis, temp{ : } );

            end % for index_object = 1:numel( grids_regular )

            % avoid cell array for single grids_regular
            if isscalar( grids_regular )
                indices_linear = indices_linear{ 1 };
            end

        end % function indices_linear = forward_index_transform( grids_regular, indices_axis )

        %------------------------------------------------------------------
        % inverse index transform
        %------------------------------------------------------------------
        function indices_axis = inverse_index_transform( grids_regular, indices_linear )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % ensure class math.grid_regular
            if ~isa( grids_regular, 'math.grid_regular' )
                errorStruct.message = 'grids_regular must be math.grid_regular!';
                errorStruct.identifier = 'inverse_index_transform:NoRegularGrids';
                error( errorStruct );
            end

            % ensure cell array for indices_linear
            if ~iscell( indices_linear )
                indices_linear = { indices_linear };
            end

            % ensure equal number of dimensions and sizes
            [ grids_regular, indices_linear ] = auxiliary.ensureEqualSize( grids_regular, indices_linear );

            %--------------------------------------------------------------
            % 2.) convert linear indices into subscripts
            %--------------------------------------------------------------
            % specify cell array for indices_axis
            indices_axis = cell( size( grids_regular ) );

            % iterate regular grids
            for index_object = 1:numel( grids_regular )

                % ensure numeric and real-valued array for indices_linear{ index_object }
                if ~( isnumeric( indices_linear{ index_object } ) && isreal( indices_linear{ index_object } ) )
                    errorStruct.message = sprintf( 'indices_linear{ %d } must be a numeric and real-valued array!', index_object );
                    errorStruct.identifier = 'inverse_index_transform:NoNumericAndRealArray';
                    error( errorStruct );
                end

                % ensure valid integers for indices_axis{ index_object }
                if ~( all( indices_linear{ index_object } == floor( indices_linear{ index_object } ), 'all' ) && all( indices_linear{ index_object } > 0, 'all' ) && all( indices_linear{ index_object } <= grids_regular( index_object ).N_points, 'all' ) )
                    errorStruct.message = sprintf( 'indices_linear{ %d } must contain integers in the correct range!', index_object );
                    errorStruct.identifier = 'inverse_index_transform:NoValidIntegers';
                    error( errorStruct );
                end

                % convert linear indices into subscripts
                temp = cell( 1, grids_regular( index_object ).N_dimensions );
                [ temp{ : } ] = ind2sub( grids_regular( index_object ).N_points_axis, indices_linear{ index_object }( : ) );
                indices_axis{ index_object } = cat( 2, temp{ : } );

            end % for index_object = 1:numel( grids_regular )

            % avoid cell array for single grids_regular
            if isscalar( grids_regular )
                indices_axis = indices_axis{ 1 };
            end

        end % function indices_axis = inverse_index_transform( grids_regular, indices_linear )

	end % methods

end % classdef grid_regular < math.grid
