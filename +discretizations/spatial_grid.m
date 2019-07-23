%
% superclass for all spatial discretizations based on grids
%
% author: Martin F. Schiffner
% date: 2019-02-17
% modified: 2019-07-15
%
classdef spatial_grid < discretizations.spatial

	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	%% properties
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	properties (SetAccess = private)

        % independent properties
% TODO: introduce class for discretized face
        grids_elements ( :, 1 ) %math.grid	% grids representing the array elements, apodization weights, and focal distances
        grid_FOV ( 1, 1 ) math.grid         % grid representing the field-of-view

	end % properties

	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	%% methods
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	methods

        %------------------------------------------------------------------
        % constructor
        %------------------------------------------------------------------
        function objects = spatial_grid( homogeneous_fluids, strs_name, grids_elements, grids_FOV )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % superclass ensures class pulse_echo_measurements.homogeneous_fluid

            % ensure cell array for grids_elements
            if ~iscell( grids_elements )
                grids_elements = { grids_elements };
            end

            % ensure class math.grid
            if ~isa( grids_FOV, 'math.grid' )
                errorStruct.message     = 'grids_FOV must be math.grid!';
                errorStruct.identifier	= 'spatial_grid:NoGrid';
                error( errorStruct );
            end

            % ensure equal number of dimensions and sizes
            auxiliary.mustBeEqualSize( homogeneous_fluids, grids_elements, grids_FOV );

            %--------------------------------------------------------------
            % 2.) create spatial discretizations based on grids
            %--------------------------------------------------------------
            % constructor of superclass
            objects@discretizations.spatial( homogeneous_fluids, strs_name );

            % iterate spatial discretizations based on grids
            for index_object = 1:numel( objects )

                % ensure class math.grid
% TODO: introduce class for discretized face
% TODO: validate N_dimensions, i.e. difference of unity!
                if ~( isa( [ grids_elements{ index_object }.grid ], 'math.grid' ) && isa( [ grids_elements{ index_object }.time_delays ], 'physical_values.time' ) )
                    errorStruct.message = sprintf( 'grids_elements{ %d } must be math.grid!', index_object );
                    errorStruct.identifier = 'spatial_grid:NoGrid';
                    error( errorStruct );
                end

                % set independent properties
                objects( index_object ).grids_elements = grids_elements{ index_object };
                objects( index_object ).grid_FOV = grids_FOV( index_object );

            end % for index_object = 1:numel( objects )

        end % function objects = spatial_grid( homogeneous_fluids, strs_name, grids_elements, grids_FOV )

        %------------------------------------------------------------------
        % compute prefactors
        %------------------------------------------------------------------
        function prefactors = compute_prefactors( spatial_grids, axes_f )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % ensure class discretizations.spatial_grid
            if ~isa( spatial_grids, 'discretizations.spatial_grid' )
                errorStruct.message = 'spatial_grids must be discretizations.spatial_grid!';
                errorStruct.identifier = 'compute_prefactors:NoSpatialGrids';
                error( errorStruct );
            end

            % ensure equal subclasses of math.grid_regular
            auxiliary.mustBeEqualSubclasses( 'math.grid_regular', spatial_grids.grid_FOV );

            % method compute_wavenumbers ensures class math.sequence_increasing

            % multiple spatial_grids / single axes_f
            if ~isscalar( spatial_grids ) && isscalar( axes_f )
                axes_f = repmat( axes_f, size( spatial_grids ) );
            end

            % single spatial_grids / multiple axes_f
            if isscalar( spatial_grids ) && ~isscalar( axes_f )
                spatial_grids = repmat( spatial_grids, size( axes_f ) );
            end

            % ensure equal number of dimensions and sizes
            auxiliary.mustBeEqualSize( spatial_grids, axes_f );

            %--------------------------------------------------------------
            % 2.) compute prefactors
            %--------------------------------------------------------------
            % specify cell array for samples
            samples = cell( size( spatial_grids ) );

            % iterate spatial discretizations based on grids
            for index_object = 1:numel( spatial_grids )

                % geometric volume element
                delta_V = spatial_grids( index_object ).grid_FOV.cell_ref.volume;

                % compute axis of complex-valued wavenumbers
                axis_k_tilde = compute_wavenumbers( spatial_grids( index_object ).homogeneous_fluid.absorption_model, axes_f( index_object ) );

                % compute samples of prefactors
                samples{ index_object } = - delta_V * axis_k_tilde.members.^2;

            end % for index_object = 1:numel( spatial_grids )

            % create signal matrices
            prefactors = discretizations.signal_matrix( axes_f, samples );

        end % function prefactors = compute_prefactors( spatial_grids, axes_f )

        %------------------------------------------------------------------
        % compute spatial transfer function
        %------------------------------------------------------------------
        function h_transfer = transfer_function( spatial_grids, axes_f, varargin )

            % internal constant
            N_points_max = 2;

            % print status
            time_start = tic;
            str_date_time = sprintf( '%04d-%02d-%02d: %02d:%02d:%02d', fix( clock ) );
            fprintf( '\t %s: computing spatial transfer function...', str_date_time );

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
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

            %--------------------------------------------------------------
            % 2.) compute spatial transfer functions
            %--------------------------------------------------------------
            % numbers of frequencies
            N_samples_f = abs( axes_f );

            % specify cell array for h_transfer
            h_transfer = cell( size( spatial_grids ) );

            % iterate spatial discretizations based on grids
            for index_grid = 1:numel( spatial_grids )

                %----------------------------------------------------------
                % a) validate indices of selected array elements
                %----------------------------------------------------------
                % ensure positive integers
                mustBeInteger( indices_element{ index_grid } );
                mustBePositive( indices_element{ index_grid } );

                % ensure that indices_element{ index_grid } does not exceed the number of array elements
                if any( indices_element{ index_grid }( : ) > numel( spatial_grids( index_grid ).grids_elements ) )
                    errorStruct.message = sprintf( 'indices_element{ %d } must not exceed the number of array elements %d!', index_grid, numel( spatial_grids( index_grid ).grids_elements ) );
                    errorStruct.identifier = 'spatial_transfer_function:InvalidIndices';
                    error( errorStruct );
                end

                %----------------------------------------------------------
                % b) compute current complex-valued wavenumbers
                %----------------------------------------------------------
                axis_k_tilde = compute_wavenumbers( spatial_grids( index_grid ).homogeneous_fluid.absorption_model, axes_f( index_grid ) );

                %----------------------------------------------------------
                % c) compute spatial transfer functions for selected array elements
                %----------------------------------------------------------
                % specify cell arrays
                h_samples = cell( size( indices_element{ index_grid } ) );

                % iterate selected array elements
                for index_selected = 1:numel( indices_element{ index_grid } )

                    %------------------------------------------------------
                    % i.) compute complex-valued apodization weights
                    %------------------------------------------------------
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

                    %------------------------------------------------------
                    % ii.) compute spatial transfer functions
                    %------------------------------------------------------
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

                end % for index_selected = 1:numel( indices_element{ index_grid } )

                % create fields
                h_transfer{ index_grid } = discretizations.field( repmat( axes_f( index_grid ), size( h_samples ) ), repmat( spatial_grids( index_grid ).grid_FOV, size( h_samples ) ), h_samples );

            end % for index_grid = 1:numel( spatial_grids )

            % avoid cell array for single spatial_grids
            if isscalar( spatial_grids )
                h_transfer = h_transfer{ 1 };
            end

            % infer and print elapsed time
            time_elapsed = toc( time_start );
            fprintf( 'done! (%f s)\n', time_elapsed );

        end % function h_transfer = transfer_function( spatial_grids, axes_f, varargin )

        %------------------------------------------------------------------
        % check for symmetry
        %------------------------------------------------------------------
        function tf = issymmetric( spatial_grids )

            % initialize results with false
            tf = false( size( spatial_grids ) );

            % iterate spatial discretizations based on grids
            for index_object = 1:numel( spatial_grids )

                % TODO: check for symmetry
            end

        end % function tf = issymmetric( spatial_grids )

	end % methods

end % classdef spatial_grid < discretizations.spatial
