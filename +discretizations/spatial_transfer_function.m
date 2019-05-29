function fields = spatial_transfer_function( spatial_grids, axes_f, varargin )
%
% compute spatial transfer function for the d-dimensional Euclidean space
%
% author: Martin F. Schiffner
% date: 2019-03-19
% modified: 2019-05-27
%

    N_points_max = 2;

    % print status
	time_start = tic;
	str_date_time = sprintf( '%04d-%02d-%02d: %02d:%02d:%02d', fix( clock ) );
	fprintf( '\t %s: computing spatial transfer function...', str_date_time );

	%----------------------------------------------------------------------
	% 1.) check arguments
	%----------------------------------------------------------------------
	% ensure class discretizations.spatial_grid
	if ~isa( spatial_grids, 'discretizations.spatial_grid' )
        errorStruct.message = 'spatial_grids must be discretizations.spatial_grid!';
        errorStruct.identifier = 'spatial_transfer_function:NoSpatialGrids';
        error( errorStruct );
    end

	% ensure class math.sequence_increasing
	if ~( isa( axes_f, 'math.sequence_increasing' ) && isa( [ axes_f.members ], 'physical_values.frequency' ) )
        errorStruct.message = 'axes_f must be math.sequence_increasing with physical_values.frequency members!';
        errorStruct.identifier = 'spatial_transfer_function:InvalidFrequencyAxis';
        error( errorStruct );
    end

	% ensure nonempty indices_element
	if nargin >= 3 && ~isempty( varargin{ 1 } )
        indices_element = varargin{ 1 };
    else
        indices_element = num2cell( ones( size( spatial_grids ) ) );
    end

	% ensure cell array for indices_element
	if ~iscell( indices_element )
        indices_element = { indices_element };
    end

	% multiple spatial_grids / single axes_f
	if ~isscalar( spatial_grids ) && isscalar( axes_f )
        axes_f = repmat( axes_f, size( spatial_grids ) );
    end

	% multiple spatial_grids / single indices_element
	if ~isscalar( spatial_grids ) && isscalar( indices_element )
        indices_element = repmat( indices_element, size( spatial_grids ) );
	end

	% ensure equal number of dimensions and sizes
	auxiliary.mustBeEqualSize( spatial_grids, axes_f, indices_element );

	%----------------------------------------------------------------------
	% 2.) compute spatial transfer functions
	%----------------------------------------------------------------------
	% numbers of frequencies
	N_samples_f = abs( axes_f );

	% complex-valued wavenumbers
    axes_k_tilde = compute_wavenumbers( [ spatial_grids.absorption_model ], axes_f );

	% specify cell array for fields
	fields = cell( size( spatial_grids ) );

	% iterate spatiospectral discretizations
	for index_object = 1:numel( spatial_grids )

        % ensure positive integers
        mustBeInteger( indices_element{ index_object } );
        mustBePositive( indices_element{ index_object } );

        % ensure that indices_element{ index_object } does not exceed number of elements
        if any( indices_element{ index_object } > numel( spatial_grids( index_object ).grids_elements ) )
            errorStruct.message = sprintf( 'indices_element{ %d } must not exceed %d!', index_object, numel( spatial_grids( index_object ).grids_elements ) );
            errorStruct.identifier = 'spatial_transfer_function:InvalidIndices';
            error( errorStruct );
        end

        % specify cell array for h_tx
        h_tx = cell( size( indices_element{ index_object } ) );

        % iterate specified elements
        for index_element = indices_element{ index_object }

            % extract discretized element
            grid_element_act = spatial_grids( index_object ).grids_elements( index_element );

            % ensure class math.grid_regular
            if ~isa( grid_element_act.grid, 'math.grid_regular' )
                errorStruct.message     = 'grid_element_act.grid must be math.grid_regular!';
                errorStruct.identifier	= 'spatial_transfer_function:NoRegularGrid';
                error( errorStruct );
            end

            % compute complex-valued apodization weights
            weights = reshape( grid_element_act.apodization .* exp( - 2j * pi * grid_element_act.time_delays * axes_f( index_object ).members' ), [ grid_element_act.grid.N_points, 1, N_samples_f( index_object ) ] );

            % initialize results with zeros
            h_tx{ index_element } = physical_values.meter( zeros( N_samples_f( index_object ), spatial_grids( index_object ).grid_FOV.N_points ) );

            % partition grid points into batches to save memory
            N_batches = ceil( grid_element_act.grid.N_points / N_points_max );
            N_points_last = grid_element_act.grid.N_points - ( N_batches - 1 ) * N_points_max;
            indices = mat2cell( (1:grid_element_act.grid.N_points), 1, [ N_points_max * ones( 1, N_batches - 1 ), N_points_last ] );

            % iterate batches
            for index_batch = 1:N_batches

                % indices of current grid points
                fprintf( '%5.1f %%', ( index_batch - 1 ) / N_batches * 1e2 );

                % compute Green's functions for specified pairs of grids and specified grid points
                temp = discretizations.greens_function( grid_element_act.grid, spatial_grids( index_object ).grid_FOV, axes_k_tilde( index_object ), indices{ index_batch } );

                % apply complex-valued apodization weights
                temp = weights( indices{ index_batch }, :, : ) .* temp;

                % integrate over aperture
                h_tx{ index_element } = h_tx{ index_element } - 2 * grid_element_act.grid.cell_ref.volume * squeeze( sum( temp, 1 ) ).';

                % indices of current grid points
                fprintf( '\b\b\b\b\b\b\b' );

            end % for index_batch = 1:N_batches

        end % for index_element = indices_element{ index_object }

        % create fields
        fields{ index_object } = discretizations.field( repmat( axes_f( index_object ), size( h_tx ) ), repmat( spatial_grids( index_object ).grid_FOV, size( h_tx ) ), h_tx );

    end % for index_object = 1:numel( spatial_grids )

	% avoid cell array for single spatial_grids
	if isscalar( spatial_grids )
        fields = fields{ 1 };
    end

    % infer and print elapsed time
	time_elapsed = toc( time_start );
	fprintf( 'done! (%f s)\n', time_elapsed );

end % function fields = spatial_transfer_function( spatial_grids, axes_f, varargin )
