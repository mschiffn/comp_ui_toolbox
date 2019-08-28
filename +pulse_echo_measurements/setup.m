%
% superclass for all pulse-echo measurement setups
%
% author: Martin F. Schiffner
% date: 2018-03-12
% modified: 2019-08-22
%
classdef setup

	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	%% properties
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	properties (SetAccess = private)

        % independent properties
        xdc_array ( 1, 1 ) transducers.array = transducers.L14_5_38             % transducer array
        homogeneous_fluid ( 1, 1 ) pulse_echo_measurements.homogeneous_fluid	% properties of the lossy homogeneous fluid
        FOV ( 1, 1 ) fields_of_view.field_of_view                               % field of view
        str_name = 'default'                                                    % name

% TODO: move to different class
        T_clk = physical_values.second( 1 / 80e6 );                     % time period of the clock signal

        % dependent properties
        intervals_tof ( :, : ) math.interval                            % lower and upper bounds on the times-of-flight

    end % properties

	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	%% methods
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	methods

        %------------------------------------------------------------------
        % constructor
        %------------------------------------------------------------------
        function objects = setup( xdc_arrays, homogeneous_fluids, FOVs, strs_name )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % ensure cell array for strs_name
            if ~iscell( strs_name )
                strs_name = { strs_name };
            end

            % ensure equal number of dimensions and sizes
            auxiliary.mustBeEqualSize( xdc_arrays, homogeneous_fluids, FOVs, strs_name );

            %--------------------------------------------------------------
            % 2.) create pulse-echo measurement setups
            %--------------------------------------------------------------
            % repeat default pulse-echo measurement setup
            objects = repmat( objects, size( xdc_arrays ) );

            % iterate pulse-echo measurement setups
            for index_object = 1:numel( objects )

                % ensure matching number of dimensions
                if xdc_arrays( index_object ).N_dimensions ~= ( FOVs( index_object ).shape.N_dimensions - 1 )
                    errorStruct.message = sprintf( 'The number of dimensions in FOVs( %d ) must exceed that in xdc_arrays( %d ) by unity!', index_object, index_object );
                    errorStruct.identifier = 'setup:DimensionMismatch';
                    error( errorStruct );
                end

                %----------------------------------------------------------
                % a) set independent properties
                %----------------------------------------------------------
                objects( index_object ).xdc_array = xdc_arrays( index_object );
                objects( index_object ).homogeneous_fluid = homogeneous_fluids( index_object );
                objects( index_object ).FOV = FOVs( index_object );
                objects( index_object ).str_name = strs_name{ index_object };

                %----------------------------------------------------------
                % b) set dependent properties
                %----------------------------------------------------------
                objects( index_object ).intervals_tof = times_of_flight( objects( index_object ) );

            end % for index_object = 1:numel( objects )

        end % function objects = setup( xdc_arrays, homogeneous_fluids, FOVs, strs_name )

        %------------------------------------------------------------------
        % lower and upper bounds on the times-of-flight
        %------------------------------------------------------------------
        function intervals_tof = times_of_flight( setups, varargin )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % ensure class pulse_echo_measurements.setup
            if ~isa( setups, 'pulse_echo_measurements.setup')
                errorStruct.message = 'setups must be pulse_echo_measurements.setup!';
                errorStruct.identifier = 'times_of_flight:NoSetups';
                error( errorStruct );
            end

            % ensure planar transducer array and FOV with orthotope shape
            indicator_array_planar = cellfun( @( x ) ~isa( x, 'transducers.array_planar' ), { setups.xdc_array } );
            indicator_FOV = cellfun( @( x ) ~isa( x.shape, 'geometry.orthotope' ), { setups.FOV } );
            if any( indicator_array_planar( : ) ) || any( indicator_FOV( : ) )
                errorStruct.message = 'Current implementation requires a planar transducer array and a FOV with orthotope shape!';
                errorStruct.identifier = 'times_of_flight:NoPlanarOrOrthotope';
                error( errorStruct );
            end

            % ensure nonempty indices_active_tx
            if nargin >= 2 && ~isempty( varargin{ 1 } )
                indices_active_tx = varargin{ 1 };
            else
                indices_active_tx = cell( size( setups ) );
                for index_setup = 1:numel( setups )
                    indices_active_tx{ index_setup } = ( 1:setups( index_setup ).xdc_array.N_elements );
                end
            end

            % ensure cell array for indices_active_tx
            if ~iscell( indices_active_tx )
                indices_active_tx = { indices_active_tx };
            end

            % ensure nonempty indices_active_rx
            if nargin >= 3 && ~isempty( varargin{ 2 } )
                indices_active_rx = varargin{ 2 };
            else
                indices_active_rx = cell( size( setups ) );
                for index_setup = 1:numel( setups )
                    indices_active_rx{ index_setup } = ( 1:setups( index_setup ).xdc_array.N_elements );
                end
            end

            % ensure cell array for indices_active_rx
            if ~iscell( indices_active_rx )
                indices_active_rx = { indices_active_rx };
            end

            % ensure equal number of dimensions and sizes
            auxiliary.mustBeEqualSize( setups, indices_active_tx, indices_active_rx );

            %--------------------------------------------------------------
            % 2.) estimate lower and upper bounds on the times-of-flight
            %--------------------------------------------------------------
            % vertices of the FOV
            pos_vertices = vertices( [ setups.FOV.shape ] );

            % ensure cell array for pos_vertices
            if ~iscell( pos_vertices )
                pos_vertices = { pos_vertices };
            end

            % specify cell array for intervals_tof
            intervals_tof = cell( size( setups ) );

            % iterate pulse-echo measurement setups
            for index_setup = 1:numel( setups )

                % initialize lower and upper bounds with zeros
                t_tof_lbs = physical_values.second( zeros( numel( indices_active_tx{ index_setup } ), numel( indices_active_rx{ index_setup } ) ) );
                t_tof_ubs = physical_values.second( zeros( numel( indices_active_tx{ index_setup } ), numel( indices_active_rx{ index_setup } ) ) );

                % iterate active tx elements
                for index_active_tx = 1:numel( indices_active_tx{ index_setup } )

                    % index of active tx element
                    index_element_tx = indices_active_tx{ index_setup }( index_active_tx );

                    % planar face of active tx element
                    face_tx = setups( index_setup ).xdc_array.aperture( index_element_tx );

                    % lower and upper bounds on the intervals
                    shape_tx_lbs = [ face_tx.shape.intervals.lb ];
                    shape_tx_ubs = [ face_tx.shape.intervals.ub ];

                    % iterate active rx elements
                    for index_active_rx = 1:numel( indices_active_rx{ index_setup } )

                        % index of active rx element
                        index_element_rx = indices_active_rx{ index_setup }( index_active_rx );

                        % planar face of active rx element
                        face_rx = setups( index_setup ).xdc_array.aperture( index_element_rx );

                        % lower and upper bounds on the intervals
                        shape_rx_lbs = [ face_rx.shape.intervals.lb ];
                        shape_rx_ubs = [ face_rx.shape.intervals.ub ];

                        % orthotope including center coordinates of prolate spheroid
                        shape_ctr_lbs = ( shape_tx_lbs + shape_rx_lbs ) / 2;
                        shape_ctr_ubs = ( shape_tx_ubs + shape_rx_ubs ) / 2;
                        shape_ctr_intervals = num2cell( math.interval( shape_ctr_lbs, shape_ctr_ubs ) );
                        shape_ctr = geometry.orthotope( shape_ctr_intervals{ : } );

% TODO: does lateral extent of FOV contain shape_ctr?
%                         if intersection( setups( index_setup ).FOV.intervals, shape_ctr_intervals )

                            % distance from center coordinates to focal points
                            dist_focus_ctr = norm( setups( index_setup ).xdc_array.positions_ctr( index_element_rx, : ) - setups( index_setup ).xdc_array.positions_ctr( index_element_tx, : ) ) / 2;

                            %----------------------------------------------
                            % a) lower bound on the time-of-flight
                            %----------------------------------------------
                            t_tof_lbs( index_active_tx, index_active_rx ) = 2 * sqrt( dist_focus_ctr^2 + setups( index_setup ).FOV.shape.intervals( end ).lb^2 ) / setups( index_setup ).homogeneous_fluid.c_avg;
                            t_tof_lbs( index_active_rx, index_active_tx ) = t_tof_lbs( index_active_tx, index_active_rx );

                            %----------------------------------------------
                            % b) upper bound on the time-of-flight
                            %----------------------------------------------
                            % determine vertices of maximum distance for lower and upper interval bounds
                            [ dist_ctr_vertices_max_lb, index_max_lb ] = max( vecnorm( [ [ shape_ctr.intervals.lb ], 0 ] - pos_vertices{ index_setup }, 2, 2 ) );
                            [ dist_ctr_vertices_max_ub, index_max_ub ] = max( vecnorm( [ [ shape_ctr.intervals.ub ], 0 ] - pos_vertices{ index_setup }, 2, 2 ) );

                            % find index and maximum distance
                            if dist_ctr_vertices_max_lb > dist_ctr_vertices_max_ub
                                index_max = index_max_lb;
                                % TODO:compute correct position
                                pos_tx = [ shape_tx_lbs, 0 ];
                                pos_rx = [ shape_rx_lbs, 0 ];
                            else
                                index_max = index_max_ub;
                                % TODO:compute correct position
                                pos_tx = [ shape_tx_ubs, 0 ];
                                pos_rx = [ shape_rx_ubs, 0 ];
                            end

                            % compute upper bound
                            t_tof_ubs( index_active_tx, index_active_rx ) = ( norm( pos_vertices{ index_setup }( index_max, : ) - pos_tx ) + norm( pos_rx - pos_vertices{ index_setup }( index_max, : ) ) ) / setups( index_setup ).homogeneous_fluid.c_avg;
                            t_tof_ubs( index_active_rx, index_active_tx ) = t_tof_ubs( index_active_tx, index_active_rx );

%                         else
                            % find vertex intersecting with smallest prolate spheroid
%                         end

                    end % for index_active_rx = 1:numel( indices_active_rx{ index_setup } )

                end % for index_active_tx = 1:numel( indices_active_tx{ index_setup } )

                % create time intervals
                intervals_tof{ index_setup } = math.interval( t_tof_lbs, t_tof_ubs );

            end % for index_setup = 1:numel( setups )

            % avoid cell array for single setup
            if isscalar( setups )
                intervals_tof = intervals_tof{ 1 };
            end

        end % function intervals_tof = times_of_flight( setups, varargin )

        %------------------------------------------------------------------
        % discretize
        %------------------------------------------------------------------
        function setups = discretize( setups, options_spatial )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % ensure class pulse_echo_measurements.setup
            if ~isa( setups, 'pulse_echo_measurements.setup')
                errorStruct.message = 'setups must be pulse_echo_measurements.setup!';
                errorStruct.identifier = 'discretize:NoSetups';
                error( errorStruct );
            end

            % ensure class discretizations.options_spatial
            if ~isa( options_spatial, 'discretizations.options_spatial')
                errorStruct.message = 'options_spatial must be discretizations.options_spatial!';
                errorStruct.identifier = 'discretize:NoOptionsSpatial';
                error( errorStruct );
            end

            % multiple setups / single options_spatial
            if ~isscalar( setups ) && isscalar( options_spatial )
                options_spatial = repmat( options_spatial, size( setups ) );
            end

            % single setups / multiple options_spatial
            if isscalar( setups ) && ~isscalar( options_spatial )
                setups = repmat( setups, size( options_spatial ) );
            end

            % ensure equal number of dimensions and sizes
            auxiliary.mustBeEqualSize( setups, options_spatial );

            %--------------------------------------------------------------
            % 2.) discretize pulse-echo measurement setups
            %--------------------------------------------------------------
            % iterate pulse-echo measurement setups
            for index_setup = 1:numel( setups )

                % discretize transducer array
                setups( index_setup ).xdc_array = discretize( setups( index_setup ).xdc_array, options_spatial( index_setup ).method_faces );

                % discretize field of view
                setups( index_setup ).FOV = discretize( setups( index_setup ).FOV, options_spatial( index_setup ).method_FOV );

            end % for index_setup = 1:numel( setups )

            %--------------------------------------------------------------
            % 3.) construct spatial discretizations
            %--------------------------------------------------------------
% TODO: vectorize 
            setup_grid_symmetric( setups )
            try
                setups = pulse_echo_measurements.setup_grid_symmetric( [ setups.xdc_array ], [ setups.homogeneous_fluid ], [ setups.FOV ], [ setups.str_name ] );
            catch
                message = 'The discrete representation of the setup is asymmetric! This significantly increases the computational costs!';
                identifier = 'discretize:AsymmetricSetup';
                warning( identifier, message );
            end

        end % function setups = discretize( setups, options_spatial )

        %------------------------------------------------------------------
        % compute prefactors
        %------------------------------------------------------------------
        function prefactors = compute_prefactors( setups, axes_f )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % ensure class pulse_echo_measurements.setup
            if ~isa( setups, 'pulse_echo_measurements.setup' )
                errorStruct.message = 'setups must be pulse_echo_measurements.setup!';
                errorStruct.identifier = 'compute_prefactors:NoSetups';
                error( errorStruct );
            end

            % method compute_wavenumbers ensures class math.sequence_increasing for axes_f

            % multiple setups / single axes_f
            if ~isscalar( setups ) && isscalar( axes_f )
                axes_f = repmat( axes_f, size( setups ) );
            end

            % single setups / multiple axes_f
            if isscalar( setups ) && ~isscalar( axes_f )
                setups = repmat( setups, size( axes_f ) );
            end

            % ensure equal number of dimensions and sizes
            auxiliary.mustBeEqualSize( setups, axes_f );

            %--------------------------------------------------------------
            % 2.) compute prefactors
            %--------------------------------------------------------------
            % specify cell array for samples
            samples = cell( size( setups ) );

            % iterate pulse-echo measurement setups
            for index_object = 1:numel( setups )

                % ensure class math.grid_regular
                if ~( isa( setups( index_object ).FOV.shape, 'geometry.orthotope_grid' ) && isa( setups( index_object ).FOV.shape.grid, 'math.grid_regular' ) )
                    errorStruct.message = 'setups must be pulse_echo_measurements.setup!';
                    errorStruct.identifier = 'compute_prefactors:NoSetups';
                    error( errorStruct );
                end

                % geometric volume element
                delta_V = setups( index_object ).FOV.shape.grid.cell_ref.volume;

                % compute axis of complex-valued wavenumbers
                axis_k_tilde = compute_wavenumbers( setups( index_object ).homogeneous_fluid.absorption_model, axes_f( index_object ) );

                % compute samples of prefactors
                samples{ index_object } = - delta_V * axis_k_tilde.members.^2;

            end % for index_object = 1:numel( setups )

            %--------------------------------------------------------------
            % 3.) create signal matrices
            %--------------------------------------------------------------
            prefactors = discretizations.signal_matrix( axes_f, samples );

        end % function prefactors = compute_prefactors( setups, axes_f )

        %------------------------------------------------------------------
        % compute spatial transfer function
        %------------------------------------------------------------------
        function h_transfer = transfer_function( setups, axes_f, varargin )

            % internal constant
            N_points_max = 1;

            % print status
            time_start = tic;
            str_date_time = sprintf( '%04d-%02d-%02d: %02d:%02d:%02d', fix( clock ) );
            fprintf( '\t %s: computing spatial transfer function... ', str_date_time );

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % ensure class pulse_echo_measurements.setup
            if ~isa( setups, 'pulse_echo_measurements.setup' )
                errorStruct.message = 'setups must be pulse_echo_measurements.setup!';
                errorStruct.identifier = 'transfer_function:NoSetups';
                error( errorStruct );
            end

% TODO: ensure discretized setup

            % ensure class math.sequence_increasing with physical_values.frequency members
            if ~( isa( axes_f, 'math.sequence_increasing' ) && all( cellfun( @( x ) isa( x, 'physical_values.frequency' ), { axes_f.members } ) ) )
                errorStruct.message = 'axes_f must be math.sequence_increasing with physical_values.frequency members!';
                errorStruct.identifier = 'transfer_function:InvalidFrequencyAxes';
                error( errorStruct );
            end

            % ensure nonempty indices_element
            if nargin >= 3 && ~isempty( varargin{ 1 } )
                indices_element = varargin{ 1 };
            else
                indices_element = num2cell( ones( size( setups ) ) );
            end

            % ensure cell array for indices_element
            if ~iscell( indices_element )
                indices_element = { indices_element };
            end

            % multiple setups / single axes_f
            if ~isscalar( setups ) && isscalar( axes_f )
                axes_f = repmat( axes_f, size( setups ) );
            end

            % multiple setups / single indices_element
            if ~isscalar( setups ) && isscalar( indices_element )
                indices_element = repmat( indices_element, size( setups ) );
            end

            % ensure equal number of dimensions and sizes
            auxiliary.mustBeEqualSize( setups, axes_f, indices_element );

            %--------------------------------------------------------------
            % 2.) compute spatial transfer functions
            %--------------------------------------------------------------
            % numbers of frequencies
            N_samples_f = abs( axes_f );

            % specify cell array for h_transfer
            h_transfer = cell( size( setups ) );

            % iterate discretized pulse-echo measurement setups
            for index_grid = 1:numel( setups )

                %----------------------------------------------------------
                % a) validate indices of selected array elements
                %----------------------------------------------------------
                % ensure positive integers
                mustBeInteger( indices_element{ index_grid } );
                mustBePositive( indices_element{ index_grid } );

                % ensure that indices_element{ index_grid } does not exceed the number of array elements
                if any( indices_element{ index_grid }( : ) > setups( index_grid ).xdc_array.N_elements )
                    errorStruct.message = sprintf( 'indices_element{ %d } must not exceed the number of array elements %d!', index_grid, setups( index_grid ).xdc_array.N_elements );
                    errorStruct.identifier = 'transfer_function:InvalidElementIndices';
                    error( errorStruct );
                end

                %----------------------------------------------------------
                % b) compute current complex-valued wavenumbers
                %----------------------------------------------------------
                axis_k_tilde = compute_wavenumbers( setups( index_grid ).homogeneous_fluid.absorption_model, axes_f( index_grid ) );

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

                    % extract discretized vibrating face
                    face_act = setups( index_grid ).xdc_array.aperture( index_element );

                    % ensure class math.grid_regular
                    if ~isa( face_act.shape.grid, 'math.grid_regular' )
                        errorStruct.message = 'face_act.shape.grid must be math.grid_regular!';
                        errorStruct.identifier = 'transfer_function:NoRegularGrid';
                        error( errorStruct );
                    end

                    % compute complex-valued wavenumbers for acoustic lens
                    axis_k_tilde_lens = compute_wavenumbers( face_act.lens.absorption_model, axes_f( index_grid ) );

                    % compute complex-valued apodization weights
                    weights = reshape( face_act.apodization .* exp( - 1j * face_act.lens.thickness * axis_k_tilde_lens.members.' ), [ face_act.shape.grid.N_points, 1, N_samples_f( index_grid ) ] );

                    %------------------------------------------------------
                    % ii.) compute spatial transfer functions
                    %------------------------------------------------------
                    % initialize samples with zeros
                    h_samples{ index_selected } = physical_values.meter( zeros( N_samples_f( index_grid ), setups( index_grid ).FOV.shape.grid.N_points ) );

                    % partition grid points into batches to save memory
                    N_batches = ceil( face_act.shape.grid.N_points / N_points_max );
                    N_points_last = face_act.shape.grid.N_points - ( N_batches - 1 ) * N_points_max;
                    indices = mat2cell( (1:face_act.shape.grid.N_points), 1, [ N_points_max * ones( 1, N_batches - 1 ), N_points_last ] );

                    % iterate batches
                    for index_batch = 1:N_batches

                        % print progress in percent
                        fprintf( '%5.1f %%', ( index_batch - 1 ) / N_batches * 1e2 );

                        % compute Green's functions for specified pairs of grids and specified grid points
                        temp = discretizations.greens_function( face_act.shape.grid, setups( index_grid ).FOV.shape.grid, axis_k_tilde, indices{ index_batch } );

                        % apply complex-valued apodization weights
                        temp = weights( indices{ index_batch }, :, : ) .* temp;

                        % integrate over aperture
                        h_samples{ index_selected } = h_samples{ index_selected } - 2 * face_act.shape.grid.cell_ref.volume * shiftdim( sum( temp, 1 ), 1 ).';

                        % erase progress in percent
                        fprintf( '\b\b\b\b\b\b\b' );

                    end % for index_batch = 1:N_batches

                end % for index_selected = 1:numel( indices_element{ index_grid } )

                %----------------------------------------------------------
                % d) create fields
                %----------------------------------------------------------
                h_transfer{ index_grid } = discretizations.field( repmat( axes_f( index_grid ), size( h_samples ) ), repmat( setups( index_grid ).FOV.shape.grid, size( h_samples ) ), h_samples );

            end % for index_grid = 1:numel( setups )

            % avoid cell array for single setups
            if isscalar( setups )
                h_transfer = h_transfer{ 1 };
            end

            % infer and print elapsed time
            time_elapsed = toc( time_start );
            fprintf( 'done! (%f s)\n', time_elapsed );

        end % function h_transfer = transfer_function( setups, axes_f, varargin )

        %------------------------------------------------------------------
        % apply anti-aliasing filter
        %------------------------------------------------------------------
        function h_transfer_aa = anti_aliasing_filter( setups, h_transfer, options_anti_aliasing, varargin )
        % apply anti-aliasing filter to
        % the spatial transfer function for the d-dimensional Euclidean space
% TODO: in-place computation in h_transfer
            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % ensure class pulse_echo_measurements.setup
            if ~isa( setups, 'pulse_echo_measurements.setup' )
                errorStruct.message = 'setups must be pulse_echo_measurements.setup!';
                errorStruct.identifier = 'anti_aliasing_filter:NoSetups';
                error( errorStruct );
            end

            % ensure class transducers.array_planar_regular_orthogonal
            indicator = cellfun( @( x ) ~isa( x, 'transducers.array_planar_regular_orthogonal' ), { setups.xdc_array } );
            if any( indicator( : ) )
                errorStruct.message = 'setups.xdc_array must be transducers.array_planar_regular_orthogonal!';
                errorStruct.identifier = 'anti_aliasing_filter:NoOrthogonalRegularPlanarArrays';
                error( errorStruct );
            end

            % ensure class discretizations.field
            if ~isa( h_transfer, 'discretizations.field' )
                errorStruct.message = 'h_transfer must be discretizations.field!';
                errorStruct.identifier = 'anti_aliasing_filter:NoFields';
                error( errorStruct );
            end

            % ensure class scattering.options.anti_aliasing
            if ~isa( options_anti_aliasing, 'scattering.options.anti_aliasing' )
                errorStruct.message = 'options_anti_aliasing must be scattering.options.anti_aliasing!';
                errorStruct.identifier = 'anti_aliasing_filter:NoAntiAliasingOptions';
                error( errorStruct );
            end

            % ensure nonempty indices_element
            if nargin >= 4 && ~isempty( varargin{ 1 } )
                indices_element = varargin{ 1 };
            else
                indices_element = num2cell( ones( size( setups ) ) );
            end

            % ensure cell array for indices_element
            if ~iscell( indices_element )
                indices_element = { indices_element };
            end

            % ensure equal number of dimensions and sizes
            auxiliary.mustBeEqualSize( setups, h_transfer, options_anti_aliasing, indices_element );

            %--------------------------------------------------------------
            % 2.) apply anti-aliasing filter
            %--------------------------------------------------------------
            % numbers of discrete frequencies
            N_samples_f = cellfun( @abs, { h_transfer.axis } );

            % specify cell array for h_samples_aa
            h_samples_aa = cell( size( setups ) );

            % iterate pulse-echo measurement setups
            for index_object = 1:numel( setups )

                % check spatial anti-aliasing filter status
                if isa( options_anti_aliasing( index_object ), 'scattering.options.anti_aliasing_off' )

                    %------------------------------------------------------
                    % a) inactive spatial anti-aliasing filter
                    %------------------------------------------------------
                    % copy spatial transfer function
                    h_samples_aa{ index_object } = h_transfer( index_object ).samples;

                else

                    %------------------------------------------------------
                    % b) active spatial anti-aliasing filter
                    %------------------------------------------------------
                    % compute lateral components of mutual unit vectors
                    e_1_minus_2 = mutual_unit_vectors( math.grid( setups( index_object ).xdc_array.positions_ctr ), h_transfer( index_object ).grid_FOV, indices_element{ index_object } );
                    e_1_minus_2 = repmat( abs( e_1_minus_2( :, :, 1:(end - 1) ) ), [ N_samples_f( index_object ), 1 ] );

                    % exclude dimensions with more than one element
                    indicator_dimensions = setups( index_object ).xdc_array.N_elements_axis > 1;
                    N_dimensions_lateral_relevant = sum( indicator_dimensions );
                    e_1_minus_2 = e_1_minus_2( :, :, indicator_dimensions );

                    % compute flag reflecting the local angular spatial frequencies
                    axis_k_tilde = compute_wavenumbers( setups( index_object ).homogeneous_fluid.absorption_model, h_transfer( index_object ).axis );
                    flag = real( axis_k_tilde.members ) .* e_1_minus_2 .* reshape( setups( index_object ).xdc_array.cell_ref.edge_lengths( indicator_dimensions ), [ 1, 1, N_dimensions_lateral_relevant ] );

                    % check type of spatial anti-aliasing filter
                    if isa( options_anti_aliasing( index_object ), 'scattering.options.anti_aliasing_boxcar' )

                        %--------------------------------------------------
                        % i.) boxcar spatial anti-aliasing filter
                        %--------------------------------------------------
                        % detect valid grid points
                        filter = all( flag < pi, 3 );

                    elseif isa( options_anti_aliasing( index_object ), 'scattering.options.anti_aliasing_raised_cosine' )

                        %--------------------------------------------------
                        % ii.) raised-cosine spatial anti-aliasing filter
                        %--------------------------------------------------
% TODO: small value of options_anti_aliasing( index_object ).roll_off_factor causes NaN
% TODO: why more conservative aliasing
                        % compute lower and upper bounds
                        flag_lb = pi * ( 1 - options_anti_aliasing( index_object ).roll_off_factor );
                        flag_ub = pi; %pi * ( 1 + options_anti_aliasing( index_object ).roll_off_factor );
                        flag_delta = flag_ub - flag_lb;

                        % detect tapered grid points
                        indicator_on = flag <= flag_lb;
                        indicator_taper = ( flag > flag_lb ) & ( flag < flag_ub );
                        indicator_off = flag >= flag_ub;

                        % compute raised-cosine function
                        flag( indicator_on ) = 1;
                        flag( indicator_taper ) = 0.5 * ( 1 + cos( pi * ( flag( indicator_taper ) - flag_lb ) / flag_delta ) );
                        flag( indicator_off ) = 0;
                        filter = prod( flag, 3 );

                    elseif isa( options_anti_aliasing( index_object ), 'scattering.options.anti_aliasing_logistic' )

                        %--------------------------------------------------
                        % iii.) logistic spatial anti-aliasing filter
                        %--------------------------------------------------
                        % compute logistic function
                        filter = prod( 1 ./ ( 1 + exp( options_anti_aliasing( index_object ).growth_rate * ( flag - pi ) ) ), 3 );

                    else

                        %--------------------------------------------------
                        % iv.) unknown spatial anti-aliasing filter
                        %--------------------------------------------------
                        errorStruct.message = sprintf( 'Class of options_anti_aliasing( %d ) is unknown!', index_object );
                        errorStruct.identifier = 'anti_aliasing_filter:UnknownOptionsClass';
                        error( errorStruct );

                    end % if isa( options_anti_aliasing( index_object ), 'scattering.options.anti_aliasing_boxcar' )

                    % apply anti-aliasing filter
                    h_samples_aa{ index_object } = h_transfer( index_object ).samples .* filter;

                end % if isa( options_anti_aliasing( index_object ), 'scattering.options.anti_aliasing_off' )

            end % for index_object = 1:numel( setups )

            %--------------------------------------------------------------
            % 3.) create fields
            %--------------------------------------------------------------
            h_transfer_aa = discretizations.field( [ h_transfer.axis ], [ h_transfer.grid_FOV ], h_samples_aa );

        end % function h_transfer_aa = anti_aliasing_filter( setups, h_transfer, options_anti_aliasing, varargin )

        %------------------------------------------------------------------
        % is discretized setup symmetric
        %------------------------------------------------------------------
        function [ tf, N_points_per_pitch_axis ] = issymmetric( setups )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % ensure class pulse_echo_measurements.setup
            if ~isa( setups, 'pulse_echo_measurements.setup')
                errorStruct.message = 'setups must be pulse_echo_measurements.setup!';
                errorStruct.identifier = 'issymmetric:NoSetups';
                error( errorStruct );
            end

            %--------------------------------------------------------------
            % 2.) check symmetric grid
            %--------------------------------------------------------------
            % initialize results with false
            tf = false( size( setups ) );
            N_points_per_pitch_axis = cell( size( setups ) );

            % iterate spatial discretizations based on grids
            for index_setup = 1:numel( setups )

                %----------------------------------------------------------
                % a) orthogonal regular planar array
                %----------------------------------------------------------
                % ensure class transducers.array_planar_regular_orthogonal
                if ~isa( setups( index_setup ).xdc_array, 'transducers.array_planar_regular_orthogonal' )
                    errorStruct.message = sprintf( 'setups( %d ).xdc_array must be transducers.array_planar_regular_orthogonal!', index_setup );
                    errorStruct.identifier = 'issymmetric:NoOrthogonalRegularPlanarArray';
                    error( errorStruct );
                end

                %----------------------------------------------------------
                % b) ensure orthotopes discretized by orthogonal regular grids
                %----------------------------------------------------------
                % ensure class geometry.orthotope_grid
                if ~isa( [ setups( index_setup ).xdc_array.aperture.shape ], 'geometry.orthotope_grid' )
                    errorStruct.message = sprintf( 'The faces of setups( %d ).xdc_array must be geometry.orthotope_grid!', index_setup );
                    errorStruct.identifier = 'issymmetric:NoGridOrthotopes';
                    error( errorStruct );
                end

                % ensure class math.grid_regular_orthogonal
                indicator = cellfun( @( x ) ~isa( x.grid, 'math.grid_regular_orthogonal' ), { setups( index_setup ).xdc_array.aperture.shape } );
                if any( indicator( : ) )
                    errorStruct.message = sprintf( 'The grids representing setups( %d ).xdc_array.aperture.shape must be math.grid_regular_orthogonal!', index_setup );
                    errorStruct.identifier = 'issymmetric:NoOrthogonalRegularGrids';
                    error( errorStruct );
                end

                % ensure class geometry.orthotope_grid
                if ~isa( setups( index_setup ).FOV.shape, 'geometry.orthotope_grid' )
                    errorStruct.message = sprintf( 'setups( %d ).FOV.shape must be geometry.orthotope_grid!', index_setup );
                    errorStruct.identifier = 'issymmetric:NoGridOrthotope';
                    error( errorStruct );
                end

                % ensure class math.grid_regular_orthogonal
                if ~isa( setups( index_setup ).FOV.shape.grid, 'math.grid_regular_orthogonal' )
                    errorStruct.message = sprintf( 'setups( %d ).FOV.shape.grid must be math.grid_regular_orthogonal!', index_setup );
                    errorStruct.identifier = 'issymmetric:NoOrthogonalRegularGrid';
                    error( errorStruct );
                end

                %----------------------------------------------------------
                % c) check lateral symmetries of the FOV about the axial axis
                %----------------------------------------------------------
% TODO: move to face or orthotope_grid
                % ensure lateral symmetries of apodization weights and thickness
                apodization = reshape( setups( index_setup ).xdc_array.aperture( 1 ).apodization, setups( index_setup ).xdc_array.aperture( 1 ).grid.N_points_axis );
                M_points_axis = ceil( ( setups( index_setup ).xdc_array.aperture( 1 ).grid.N_points_axis - 1 ) / 2 );
%                 apodization( :, 1 ) - apodization( :, end )

                % ensure lateral symmetries of the FOV about the axial axis
                FOV_pos_ctr = 2 * setups( index_setup ).FOV.shape.grid.offset_axis( 1:(end - 1) ) + ( setups( index_setup ).FOV.shape.grid.N_points_axis( 1:(end - 1) ) - 1 ) .* setups( index_setup ).FOV.shape.grid.cell_ref.edge_lengths( 1:(end - 1) );
                if any( abs( double( FOV_pos_ctr ) ) > eps( 0 ) )
                    errorStruct.message = 'Symmetric spatial grid requires the lateral symmetries of the FOV about the axial axis!';
                    errorStruct.identifier = 'issymmetric:NoSymmetry';
                    error( errorStruct );
                end

                %----------------------------------------------------------
                % d) TODO: check minimal # of lateral grid points
                %----------------------------------------------------------
                % minimum number of grid points on x-axis [ FOV_pos_x(1) <= XDC_pos_ctr_x(1) ]
                % 1.) x-coordinates of grid points coincide with centroids of vibrating faces:
                %   a) XDC_N_elements:odd, FOV_N_points_axis(1):odd  / b) XDC_N_elements:even, FOV_N_points_axis(1):odd, factor_interp_tx:even / c) XDC_N_elements:even, FOV_N_points_axis(1):even, factor_interp_tx:odd
                %   N_lattice_axis_x_lb = ( XDC_N_elements - 1 ) * factor_interp_tx + 1;
                % 2.) x-coordinates of grid points do not coincide with centroids of vibrating faces:
                %   a) XDC_N_elements:odd, FOV_N_points_axis(1):even / b) XDC_N_elements:even, FOV_N_points_axis(1):odd, factor_interp_tx:odd  / c) XDC_N_elements:even, FOV_N_points_axis(1):even, factor_interp_tx:even
                %   N_lattice_axis_x_lb = ( XDC_N_elements - 1 ) * factor_interp_tx;
                % if mod( XDC_N_elements, 2 ) ~= 0
                % 	% 1.) odd number of physical transducer elements
                % 	N_lattice_axis_x_lb = ( XDC_N_elements - 1 ) * factor_interp_tx + mod( FOV_N_points_axis(1), 2 );
                % else
                %     % 2.) even number of physical transducer elements
                %     N_lattice_axis_x_lb = ( XDC_N_elements - 1 ) * factor_interp_tx + mod( FOV_N_points_axis(1), 2 ) * ( 1 - mod( factor_interp_tx, 2 ) ) + ( 1 - mod( FOV_N_points_axis(1), 2 ) ) * mod( factor_interp_tx, 2 );
                % end
                % N_lattice_axis_x_symmetry_left = ( FOV_N_points_axis( 1 ) - N_lattice_axis_x_lb ) / 2;

                % % check excess number of grid points on x-axis
                % if N_lattice_axis_x_symmetry_left < 0
                % 	errorStruct.message	   = sprintf( 'Number of grid points along the r1-axis must be equal to or greater than %d!\n', N_lattice_axis_x_lb );
                % 	errorStruct.identifier = sprintf( '%s:DiscretizationError', NAME );
                % 	error( errorStruct );
                % end
                % % assertion: N_lattice_axis_x_symmetry_left >= 0

                %----------------------------------------------------------
                % e) lateral spacing is an integer fraction of the element pitch
                %    => translational invariance by shifts of factor_interp_tx points
                %----------------------------------------------------------
                N_points_per_pitch_axis{ index_setup } = setups( index_setup ).xdc_array.cell_ref.edge_lengths ./ setups( index_setup ).FOV.shape.grid.cell_ref.edge_lengths( 1:( end - 1 ) );
                if any( abs( N_points_per_pitch_axis{ index_setup } - round( N_points_per_pitch_axis{ index_setup } ) ) > eps( round( N_points_per_pitch_axis{ index_setup } ) ) )
                    errorStruct.message = 'Symmetric discretized setup requires the lateral spacings of the grid points in the FOV to be integer fractions of the element pitch!';
                    errorStruct.identifier = 'issymmetric:NoIntegerFraction';
                    error( errorStruct );
                end

                % declare discretized setup to be symmetric
                tf( index_setup ) = true;
                N_points_per_pitch_axis{ index_setup } = round( N_points_per_pitch_axis{ index_setup } );

            end % for index_setup = 1:numel( setups )

            % avoid cell array for single setups
            if isscalar( setups )
                N_points_per_pitch_axis = N_points_per_pitch_axis{ 1 };
            end

        end % function tf = issymmetric( setups )

        %
        function setup_grid_symmetric( setups )
        end

    end % methods

end % classdef setup
