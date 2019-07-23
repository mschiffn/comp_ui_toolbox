function [ h_transfer, h_transfer_aa ] = spatial_transfer_function( spatial_grids, axes_f, varargin )
%
% compute spatial transfer function for the d-dimensional Euclidean space
%
% author: Martin F. Schiffner
% date: 2019-03-19
% modified: 2019-07-10
%

% TODO: make method of spatial_grid

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

	% ensure class math.sequence_increasing with physical_values.frequency members
	if ~( isa( axes_f, 'math.sequence_increasing' ) && all( cellfun( @( x ) isa( x, 'physical_values.frequency' ), { axes_f.members } ) ) )
        errorStruct.message = 'axes_f must be math.sequence_increasing with physical_values.frequency members!';
        errorStruct.identifier = 'spatial_transfer_function:InvalidFrequencyAxes';
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

	% specify cell arrays
	h_transfer = cell( size( spatial_grids ) );
    h_transfer_aa = cell( size( spatial_grids ) );

	% iterate spatial discretizations based on grids
	for index_grid = 1:numel( spatial_grids )

        %------------------------------------------------------------------
        % a) validate indices of selected array elements
        %------------------------------------------------------------------
        % ensure positive integers
        mustBeInteger( indices_element{ index_grid } );
        mustBePositive( indices_element{ index_grid } );

        % ensure that indices_element{ index_grid } does not exceed the number of array elements
        if any( indices_element{ index_grid }( : ) > numel( spatial_grids( index_grid ).grids_elements ) )
            errorStruct.message = sprintf( 'indices_element{ %d } must not exceed the number of array elements %d!', index_grid, numel( spatial_grids( index_grid ).grids_elements ) );
            errorStruct.identifier = 'spatial_transfer_function:InvalidIndices';
            error( errorStruct );
        end

        %------------------------------------------------------------------
        % b) compute current complex-valued wavenumbers
        %------------------------------------------------------------------
        axis_k_tilde = compute_wavenumbers( spatial_grids( index_grid ).homogeneous_fluid.absorption_model, axes_f( index_grid ) );

        %------------------------------------------------------------------
        % c) compute spatial transfer functions for selected array elements
        %------------------------------------------------------------------
        % specify cell arrays
        h_samples = cell( size( indices_element{ index_grid } ) );
        h_samples_aa = cell( size( indices_element{ index_grid } ) );

        % iterate selected array elements
        for index_selected = 1:numel( indices_element{ index_grid } )

            %--------------------------------------------------------------
            % i.) compute complex-valued apodization weights
            %--------------------------------------------------------------
            % index of current array element
            index_element = indices_element{ index_grid }( index_selected );

            % extract discretized array element
            grid_element_act = spatial_grids( index_grid ).grids_elements( index_element );

            % ensure class math.grid_regular
            if ~isa( grid_element_act.grid, 'math.grid_regular' )
                errorStruct.message = 'grid_element_act.grid must be math.grid_regular!';
                errorStruct.identifier = 'spatial_transfer_function:NoRegularGrid';
                error( errorStruct );
            end

            % compute complex-valued apodization weights
            weights = reshape( grid_element_act.apodization .* exp( - 2j * pi * grid_element_act.time_delays * axes_f( index_grid ).members' ), [ grid_element_act.grid.N_points, 1, N_samples_f( index_grid ) ] );

            %--------------------------------------------------------------
            % ii.) compute spatial transfer functions
            %--------------------------------------------------------------
            % initialize samples with zeros
            h_samples{ index_selected } = physical_values.meter( zeros( N_samples_f( index_grid ), spatial_grids( index_grid ).grid_FOV.N_points ) );

            % partition grid points into batches to save memory
            N_batches = ceil( grid_element_act.grid.N_points / N_points_max );
            N_points_last = grid_element_act.grid.N_points - ( N_batches - 1 ) * N_points_max;
            indices = mat2cell( (1:grid_element_act.grid.N_points), 1, [ N_points_max * ones( 1, N_batches - 1 ), N_points_last ] );

            % iterate batches
            for index_batch = 1:N_batches

                % print progress in percent
                fprintf( '%5.1f %%', ( index_batch - 1 ) / N_batches * 1e2 );

                % compute Green's functions for specified pairs of grids and specified grid points
                temp = discretizations.greens_function( grid_element_act.grid, spatial_grids( index_grid ).grid_FOV, axis_k_tilde, indices{ index_batch } );

                % apply complex-valued apodization weights
                temp = weights( indices{ index_batch }, :, : ) .* temp;

                % integrate over aperture
                h_samples{ index_selected } = h_samples{ index_selected } - 2 * grid_element_act.grid.cell_ref.volume * shiftdim( sum( temp, 1 ), 1 ).';

                % erase progress in percent
                fprintf( '\b\b\b\b\b\b\b' );

            end % for index_batch = 1:N_batches

            %--------------------------------------------------------------
            % iii.) apply anti-aliasing filter
            %--------------------------------------------------------------
% TODO: own function discretizations.anti_aliasing_filter?
% regular array: center of array elements, element_pitch
% grid_FOV
% TODO: center of array elements?
            e_1_minus_2 = mutual_unit_vectors( math.grid( sum( grid_element_act.grid.positions ) / grid_element_act.grid.N_points ), spatial_grids( index_grid ).grid_FOV, 1 );
            e_1_minus_2 = repmat( abs( e_1_minus_2( :, :, 1:(end - 1) ) ), [ N_samples_f( index_grid ), 1 ] );
% TODO: element pitch is vector!
            element_pitch = physical_values.meter( 304.8e-6 );
            flag = real( axis_k_tilde.members ) .* e_1_minus_2 * element_pitch;

            indicator_no_aliasing = all( flag < pi, 3 );
            r = 1;
            indicator_taper = flag >= pi * ( 1 - r );

            flag( ~indicator_taper ) = 1;
            flag( indicator_taper ) = cos( ( flag( indicator_taper ) - pi * ( 1 - r ) ) / ( 2 * r ) );
            flag = indicator_no_aliasing .* prod( flag, 3 );

            h_samples_aa{ index_selected } = h_samples{ index_selected } .* flag;

        end % for index_selected = 1:numel( indices_element{ index_grid } )

        % create fields
        h_transfer{ index_grid } = discretizations.field( repmat( axes_f( index_grid ), size( h_samples ) ), repmat( spatial_grids( index_grid ).grid_FOV, size( h_samples ) ), h_samples );
        h_transfer_aa{ index_grid } = discretizations.field( repmat( axes_f( index_grid ), size( h_samples_aa ) ), repmat( spatial_grids( index_grid ).grid_FOV, size( h_samples_aa ) ), h_samples_aa );

    end % for index_grid = 1:numel( spatial_grids )

	% avoid cell array for single spatial_grids
	if isscalar( spatial_grids )
        h_transfer = h_transfer{ 1 };
        h_transfer_aa = h_transfer_aa{ 1 };
    end

    % infer and print elapsed time
	time_elapsed = toc( time_start );
	fprintf( 'done! (%f s)\n', time_elapsed );

end % function [ h_transfer, h_transfer_aa ] = spatial_transfer_function( spatial_grids, axes_f, varargin )
