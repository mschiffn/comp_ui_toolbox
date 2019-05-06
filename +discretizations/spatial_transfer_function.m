function fields = spatial_transfer_function( spatial_grid, spectral_points, varargin )
% spatial transfer function for the d-dimensional Euclidean space
%
% author: Martin F. Schiffner
% date: 2019-03-19
% modified: 2019-05-04

    N_points_max = 4;

	%----------------------------------------------------------------------
	% 1.) check arguments
	%----------------------------------------------------------------------
	% ensure class discretizations.spatial_grid (scalar)
	if ~( isa( spatial_grid, 'discretizations.spatial_grid' ) && isscalar( spatial_grid ) )
        errorStruct.message = 'spatial_grid must be a single discretizations.spatial_grid!';
        errorStruct.identifier = 'spatial_transfer_function:NoSpatialGrid';
        error( errorStruct );
    end

	% ensure class discretizations.spectral_points
	if ~isa( spectral_points, 'discretizations.spectral_points' )
        errorStruct.message = 'spectral_points must be discretizations.spectral_points!';
        errorStruct.identifier = 'spatial_transfer_function:NoSpectralPoints';
        error( errorStruct );
    end

	% ensure nonempty indices_element
	if nargin >= 3 && ~isempty( varargin{ 1 } )
        indices_element = varargin{ 1 };
    else
        indices_element = num2cell( ones( size( spectral_points ) ) );
    end

	% ensure cell array for indices_element
	if ~iscell( indices_element )
        indices_element = { indices_element };
    end

	% multiple spectral_points / single indices_element
	if ~isscalar( spectral_points ) && isscalar( indices_element )
        indices_element = repmat( indices_element, size( spectral_points ) );
	end

	% ensure equal number of dimensions and sizes
	auxiliary.mustBeEqualSize( spectral_points, indices_element );

    %----------------------------------------------------------------------
	% 2.) compute spatial transfer functions
	%----------------------------------------------------------------------
	% specify cell array for fields
    fields = cell( size( spectral_points ) );

    % iterate spectral discretizations based on pointwise sampling
    for index_object = 1:numel( spectral_points )

        % ensure positive integers
        mustBeInteger( indices_element{ index_object } );
        mustBePositive( indices_element{ index_object } );

        % ensure that indices_element{ index_object } does not exceed number of elements
        if any( indices_element{ index_object } > numel( spatial_grid.grids_elements ) )
            errorStruct.message = sprintf( 'indices_element{ %d } must not exceed %d!', index_object, numel( spatial_grid.grids_elements ) );
            errorStruct.identifier = 'spatial_transfer_function:InvalidIndices';
            error( errorStruct );
        end

        % extract axis of unique frequencies
        axis_f = spectral_points( index_object ).tx_unique.excitation_voltages.axis;
        axis_k_tilde = spectral_points( index_object ).axis_k_tilde_unique;
        N_samples_f = abs( axis_f );

        % iterate specified elements
        for index_element = indices_element{ index_object }

            % extract discretized element
            grid_element_act = spatial_grid.grids_elements( index_element );

            % ensure class math.grid_regular
            if ~isa( grid_element_act.grid, 'math.grid_regular' )
                errorStruct.message     = 'grid_element_act.grid must be math.grid_regular!';
                errorStruct.identifier	= 'spatial_transfer_function:NoRegularGrid';
                error( errorStruct );
            end

            % compute complex-valued apodization weights
            weights = reshape( grid_element_act.apodization .* exp( - 2j * pi * grid_element_act.time_delays * axis_f.members ), [ grid_element_act.grid.N_points, 1, N_samples_f ] );

            % initialize results with zeros
            h_tx{ index_object } = physical_values.meter( zeros( spatial_grid.grid_FOV.N_points, N_samples_f ) );

            % partition grid points into batches to save memory
            N_batches = ceil( grid_element_act.grid.N_points / N_points_max );
            N_points_last = grid_element_act.grid.N_points - ( N_batches - 1 ) * N_points_max;
            indices = mat2cell( (1:grid_element_act.grid.N_points), 1, [ N_points_max * ones( 1, N_batches - 1 ), N_points_last ] );

            % iterate batches
            for index_batch = 1:N_batches

                % indices of current grid points
                disp( indices{ index_batch } );

                % compute Green's functions for specified pairs of grids and specified grid points
                temp = discretizations.greens_function( grid_element_act.grid, spatial_grid.grid_FOV, axis_k_tilde, indices{ index_batch } );

                % apply complex-valued apodization weights
                temp = weights( indices{ index_batch }, :, : ) .* temp;

                % integrate over aperture
                h_tx{ index_object } = h_tx{ index_object } - 2 * grid_element_act.grid.cell_ref.volume * squeeze( sum( temp, 1 ) );

            end % for index_batch = 1:N_batches

        end % for index_element = indices_element{ index_object }

        % create fields
        fields{ index_object } = discretizations.field( axis_f, spatial_grid.grid_FOV, h_tx );

    end % for index_object = 1:numel( spectral_points )

    % avoid cell array for single transducer control setting
	if isscalar( spectral_points )
        fields = fields{ 1 };
    end

end % function fields = spatial_transfer_function( spatial_grid, spectral_points, varargin )
