%
% superclass for all pulse-echo measurement setups
%
% author: Martin F. Schiffner
% date: 2018-03-12
% modified: 2020-02-20
%
classdef setup

	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	%% properties
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	properties (SetAccess = private)

        % independent properties
        xdc_array ( 1, 1 ) scattering.sequences.setups.transducers.array = scattering.sequences.setups.transducers.L14_5_38	% transducer array
        homogeneous_fluid ( 1, 1 ) scattering.sequences.setups.materials.homogeneous_fluid	% properties of the lossy homogeneous fluid
        FOV ( 1, 1 ) scattering.sequences.setups.fields_of_view.field_of_view               % field of view
        str_name = 'default'                                                                % name

% TODO: move to different class!
        T_clk = physical_values.second( 1 / 80e6 );                   % time period of the clock signal
%         T_clk = physical_values.second( ( 1 / 20832000 ) / 12 );        % time period of the clock signal
%         T_clk = physical_values.second( 1 / (12 * 20832000) );          % time period of the clock signal

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
            % ensure class scattering.sequences.setups.setup
            if ~isa( setups, 'scattering.sequences.setups.setup')
                errorStruct.message = 'setups must be scattering.sequences.setups.setup!';
                errorStruct.identifier = 'times_of_flight:NoSetups';
                error( errorStruct );
            end

            % ensure planar transducer array and FOV with orthotope shape
% TODO: ensure orthotopic shapes of the faces!
            indicator_array_planar = cellfun( @( x ) ~isa( x, 'scattering.sequences.setups.transducers.array_planar' ), { setups.xdc_array } );
%             indicator_faces_orthotope = cellfun( @( x ) ~isa( x.aperture.shape, 'scattering.sequences.setups.transducers.array_planar' ), { setups.xdc_array } );
            indicator_FOV = cellfun( @( x ) ~isa( x.shape, 'scattering.sequences.setups.geometry.orthotope' ), { setups.FOV } );
            if any( indicator_array_planar( : ) ) || any( indicator_FOV( : ) )
                errorStruct.message = 'Current implementation of method times_of_flight requires a planar transducer array and a FOV with orthotope shape!';
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
            % specify cell array for intervals_tof
            intervals_tof = cell( size( setups ) );

            % iterate pulse-echo measurement setups
            for index_setup = 1:numel( setups )

                % print status
                time_start = tic;
                str_date_time = sprintf( '%04d-%02d-%02d: %02d:%02d:%02d', fix( clock ) );
                fprintf( '\t %s: computing lower and upper bounds on the times-of-flight...', str_date_time );

                % initialize lower and upper bounds with zeros
                t_tof_lbs = physical_values.second( zeros( numel( indices_active_tx{ index_setup } ), numel( indices_active_rx{ index_setup } ) ) );
                t_tof_ubs = physical_values.second( zeros( numel( indices_active_tx{ index_setup } ), numel( indices_active_rx{ index_setup } ) ) );

                % check for symmetric bounds
                if isequal( indices_active_tx{ index_setup }, indices_active_rx{ index_setup } )
                    flag_symmetric = true;
                else
                    flag_symmetric = false;
                end

                % lower and upper bounds on the intervals
                FOV_lbs = [ setups( index_setup ).FOV.shape.intervals.lb ];
                FOV_ubs = [ setups( index_setup ).FOV.shape.intervals.ub ];

                % initialize elapsed times with zero
                seconds_per_tx = zeros( 1, numel( indices_active_tx{ index_setup } ) );

                % iterate active tx elements
                for index_active_tx = 1:numel( indices_active_tx{ index_setup } )

                    % print progress in percent
                    if index_active_tx > 1
                        N_bytes = fprintf( '%5.1f %% (elapsed: %d min. | remaining: %d min. | mean: %.2f s | last: %.2f s)', ( index_active_tx - 1 ) / numel( indices_active_tx{ index_setup } ) * 1e2, round( toc( time_start ) / 60 ), round( ( numel( indices_active_tx{ index_setup } ) - index_active_tx + 1 ) * mean( seconds_per_tx( 1:(index_active_tx - 1) ) ) / 60 ), mean( seconds_per_tx( 1:(index_active_tx - 1) ) ), seconds_per_tx( index_active_tx - 1 ) );
                    else
                        N_bytes = fprintf( '%5.1f %% (elapsed: %d min.)', 0, round( toc( time_start ) / 60 ) );
                    end

                    % start time measurement per tx
                    time_tx_start = tic;

                    %------------------------------------------------------
                    % extract lower and upper bounds on tx face
                    %------------------------------------------------------
                    % index of active tx element
                    index_element_tx = indices_active_tx{ index_setup }( index_active_tx );

                    % planar face of active tx element
                    face_tx = setups( index_setup ).xdc_array.aperture( index_element_tx );

                    % lower and upper bounds on the intervals
                    face_tx_lbs = [ face_tx.shape.intervals.lb ];
                    face_tx_ctr = setups( index_setup ).xdc_array.positions_ctr( index_element_tx, : );
                    face_tx_ubs = [ face_tx.shape.intervals.ub ];

                    % set upper bound for iteration
                    if flag_symmetric
                        index_active_rx_ub = index_active_tx;
                    else
                        index_active_rx_ub = numel( indices_active_rx{ index_setup } );
                    end

                    % iterate active rx elements
                    for index_active_rx = 1:index_active_rx_ub

                        %--------------------------------------------------
                        % extract lower and upper bounds on rx face
                        %--------------------------------------------------
                        % index of active rx element
                        index_element_rx = indices_active_rx{ index_setup }( index_active_rx );

                        % planar face of active rx element
                        face_rx = setups( index_setup ).xdc_array.aperture( index_element_rx );

                        % lower and upper bounds on the intervals
                        face_rx_lbs = [ face_rx.shape.intervals.lb ];
                        face_rx_ctr = setups( index_setup ).xdc_array.positions_ctr( index_element_rx, : );
                        face_rx_ubs = [ face_rx.shape.intervals.ub ];

                        %--------------------------------------------------
                        % lower and upper bounds on the center coordinates of the prolate spheroid
                        %--------------------------------------------------
                        spheroid_ctr_lbs = ( face_tx_lbs + face_rx_lbs ) / 2;
                        spheroid_ctr_ubs = ( face_tx_ubs + face_rx_ubs ) / 2;

                        % initialize positions w/ zeros
                        position_tx = physical_values.meter( zeros( 1, setups( index_setup ).FOV.shape.N_dimensions ) );
                        position_rx = physical_values.meter( zeros( 1, setups( index_setup ).FOV.shape.N_dimensions ) );
                        position_FOV = physical_values.meter( zeros( 1, setups( index_setup ).FOV.shape.N_dimensions ) );

                        % display results
%                         figure(1);
%                         plot( face_tx_lbs( 1 ), face_tx_lbs( 2 ) );
%                         line( [ face_tx_lbs( 1 ), face_tx_ubs( 1 ), face_tx_lbs( 1 ), face_tx_lbs( 1 ); face_tx_lbs( 1 ), face_tx_ubs( 1 ), face_tx_ubs( 1 ), face_tx_ubs( 1 ) ], [ face_tx_lbs( 2 ), face_tx_lbs( 2 ), face_tx_lbs( 2 ), face_tx_ubs( 2 ); face_tx_ubs( 2 ), face_tx_ubs( 2 ), face_tx_lbs( 2 ), face_tx_ubs( 2 ) ], 'Color', 'b' );
%                         line( [ face_rx_lbs( 1 ), face_rx_ubs( 1 ), face_rx_lbs( 1 ), face_rx_lbs( 1 ); face_rx_lbs( 1 ), face_rx_ubs( 1 ), face_rx_ubs( 1 ), face_rx_ubs( 1 ) ], [ face_rx_lbs( 2 ), face_rx_lbs( 2 ), face_rx_lbs( 2 ), face_rx_ubs( 2 ); face_rx_ubs( 2 ), face_rx_ubs( 2 ), face_rx_lbs( 2 ), face_rx_ubs( 2 ) ], 'Color', 'r' );
%                         line( [ FOV_lbs( 1 ), FOV_ubs( 1 ), FOV_lbs( 1 ), FOV_lbs( 1 ); FOV_lbs( 1 ), FOV_ubs( 1 ), FOV_ubs( 1 ), FOV_ubs( 1 ) ], [ FOV_lbs( 2 ), FOV_lbs( 2 ), FOV_lbs( 2 ), FOV_ubs( 2 ); FOV_ubs( 2 ), FOV_ubs( 2 ), FOV_lbs( 2 ), FOV_ubs( 2 ) ], 'Color', 'g' );
%                         hold on;

                        %--------------------------------------------------
                        % a) lower bound on the time-of-flight
                        %--------------------------------------------------
                        % center position
                        position_ctr = 0.5 * ( min( face_tx_ubs, face_rx_ubs ) + max( face_tx_lbs, face_rx_lbs ) );

                        % define cases
                        indicator_1 = FOV_ubs( 1:(end - 1) ) <= min( face_tx_lbs, face_rx_lbs );
                        indicator_2 = ( FOV_ubs( 1:(end - 1) ) > min( face_tx_lbs, face_rx_lbs ) ) & ( FOV_ubs( 1:(end - 1) ) <= min( face_tx_ubs, face_rx_ubs ) );
                        indicator_3 = ( FOV_ubs( 1:(end - 1) ) > min( face_tx_ubs, face_rx_ubs ) ) & ( FOV_ubs( 1:(end - 1) ) <= position_ctr );
                        indicator_4 = ( FOV_lbs( 1:(end - 1) ) <= position_ctr ) & ( FOV_ubs( 1:(end - 1) ) > position_ctr );
                        indicator_5 = ( FOV_lbs( 1:(end - 1) ) > position_ctr ) & ( FOV_lbs( 1:(end - 1) ) <= max( face_tx_lbs, face_rx_lbs ) );
                        indicator_6 = ( FOV_lbs( 1:(end - 1) ) > max( face_tx_lbs, face_rx_lbs ) ) & ( FOV_lbs( 1:(end - 1) ) < max( face_tx_ubs, face_rx_ubs ) );
                        indicator_7 = FOV_lbs( 1:(end - 1) ) >= max( face_tx_ubs, face_rx_ubs );

                        % relative position of tx and rx faces
                        indicator_tx_left = face_tx_ubs < face_rx_lbs;
                        indicator_tx_right = face_rx_ubs < face_tx_lbs;
                        indicator_tx_equal = ~indicator_tx_left & ~ indicator_tx_right;
                        indicator_FOV_left = indicator_1 | indicator_2 | indicator_3;
                        indicator_FOV_right = indicator_5 | indicator_6 | indicator_7;

                        % tx positions
                        position_tx( indicator_1 ) = face_tx_lbs( indicator_1 );
                        position_tx( indicator_2 ) = ( indicator_tx_left( indicator_2 ) + indicator_tx_equal( indicator_2 ) ) .* FOV_ubs( [ indicator_2, false ] ) + ...
                                                       indicator_tx_right( indicator_2 ) .* face_tx_lbs( indicator_2 );
                        position_tx( indicator_3 ) = indicator_tx_left( indicator_3 ) .* face_tx_ubs( indicator_3 ) + ...
                                                     indicator_tx_right( indicator_3 ) .* face_tx_lbs( indicator_3 );
                        position_tx( indicator_4 ) = indicator_tx_left( indicator_4 ) .* face_tx_ubs( indicator_4 ) + ...
                                                     indicator_tx_right( indicator_4 ) .* face_tx_lbs( indicator_4 ) + ...
                                                     indicator_tx_equal( indicator_4 ) .* position_ctr( indicator_4 );
                        position_tx( indicator_5 ) = indicator_tx_left( indicator_5 ) .* face_tx_ubs( indicator_5 ) + ...
                                                     indicator_tx_right( indicator_5 ) .* face_tx_lbs( indicator_5 );
                        position_tx( indicator_6 ) = indicator_tx_left( indicator_6 ) .* face_tx_ubs( indicator_6 ) + ...
                                                   ( indicator_tx_right( indicator_6 ) + indicator_tx_equal( indicator_6 ) ) .* FOV_lbs( [ indicator_6, false ] );
                        position_tx( indicator_7 ) = face_tx_ubs( indicator_7 );

                        % rx positions
                        position_rx( indicator_1 ) = face_rx_lbs( indicator_1 );
                        position_rx( indicator_2 ) =   indicator_tx_left( indicator_2 ) .* face_rx_lbs( indicator_2 ) + ...
                                                     ( indicator_tx_right( indicator_2 ) + indicator_tx_equal( indicator_2 ) ) .* FOV_ubs( [ indicator_2, false ] );
                        position_rx( indicator_3 ) = indicator_tx_left( indicator_3 ) .* face_rx_lbs( indicator_3 ) + ...
                                                     indicator_tx_right( indicator_3 ) .* face_rx_ubs( indicator_3 );
                        position_rx( indicator_4 ) = indicator_tx_left( indicator_4 ) .* face_rx_lbs( indicator_4 ) + ...
                                                     indicator_tx_right( indicator_4 ) .* face_rx_ubs( indicator_4 ) + ...
                                                     indicator_tx_equal( indicator_4 ) .* position_ctr( indicator_4 );
                        position_rx( indicator_5 ) = indicator_tx_left( indicator_5 ) .* face_rx_lbs( indicator_5 ) + ...
                                                     indicator_tx_right( indicator_5 ) .* face_rx_ubs( indicator_5 );
                        position_rx( indicator_6 ) = ( indicator_tx_left( indicator_6 ) + indicator_tx_equal( indicator_6 ) ) .* FOV_lbs( [ indicator_6, false ] ) + ...
                                                      indicator_tx_right( indicator_6 ) .* face_rx_ubs( indicator_6 );
                        position_rx( indicator_7 ) = face_rx_ubs( indicator_7 );

                        % FOV positions
                        position_FOV( indicator_FOV_left ) = FOV_ubs( [ indicator_FOV_left, false ] );
                        position_FOV( indicator_4 ) = position_ctr( indicator_4 );
                        position_FOV( indicator_FOV_right ) = FOV_lbs( [ indicator_FOV_right, false ] );
                        position_FOV( end ) = FOV_lbs( end );

                        % lower bound on the time-of-flight
                        t_tof_lbs( index_active_tx, index_active_rx ) = ( norm( position_FOV - position_tx ) + norm( position_rx - position_FOV ) ) / setups( index_setup ).homogeneous_fluid.c_avg;

                        % display results
%                         plot( position_tx( 1 ), position_tx( 2 ), '+', 'Color', 'b' );
%                         plot( position_rx( 1 ), position_rx( 2 ), '+', 'Color', 'r' );
%                         plot( position_FOV( 1 ), position_FOV( 2 ), '+', 'Color', 'g' );

                        %--------------------------------------------------
                        % b) upper bound on the time-of-flight
                        %--------------------------------------------------
                        % define cases
                        indicator_FOV_lbs = abs( FOV_lbs( 1:(end - 1) ) - spheroid_ctr_ubs ) >= abs( FOV_ubs( 1:(end - 1) ) - spheroid_ctr_lbs );

                        indicator_1 = indicator_FOV_lbs & FOV_lbs( 1:(end - 1) ) <= min( face_tx_ctr, face_rx_ctr );
                        indicator_2 = indicator_FOV_lbs & FOV_lbs( 1:(end - 1) ) > min( face_tx_ctr, face_rx_ctr );
                        indicator_3 = ~indicator_FOV_lbs & FOV_ubs( 1:(end - 1) ) < max( face_tx_ctr, face_rx_ctr );
                        indicator_4 = ~indicator_FOV_lbs & FOV_ubs( 1:(end - 1) ) >= max( face_tx_ctr, face_rx_ctr );

                        indicator_5 = indicator_2 | indicator_3;

                        % tx positions
                        position_tx( indicator_1 ) = face_tx_ubs( indicator_1 );
                        position_tx( indicator_5 ) = indicator_tx_left( indicator_5 ) .* face_tx_lbs( indicator_5 ) + indicator_tx_right( indicator_5 ) .* face_tx_ubs( indicator_5 );
                        position_tx( indicator_4 ) = face_tx_lbs( indicator_4 );

                        % rx positions
                        position_rx( indicator_1 ) = face_rx_ubs( indicator_1 );
                        position_rx( indicator_5 ) = indicator_tx_left( indicator_5 ) .* face_rx_ubs( indicator_5 ) + indicator_tx_right( indicator_5 ) .* face_rx_lbs( indicator_5 );
                        position_rx( indicator_4 ) = face_rx_lbs( indicator_4 );

                        % FOV positions
                        position_FOV( indicator_FOV_lbs ) = FOV_lbs( [ indicator_FOV_lbs, false ] );
                        position_FOV( ~indicator_FOV_lbs ) = FOV_ubs( [ ~indicator_FOV_lbs, false ] );
                        position_FOV( end ) = FOV_ubs( end );

                        % upper bound on the time-of-flight
                        t_tof_ubs( index_active_tx, index_active_rx ) = ( norm( position_FOV - position_tx ) + norm( position_rx - position_FOV ) ) / setups( index_setup ).homogeneous_fluid.c_avg;

                        % display results
%                         plot( position_tx( 1 ), position_tx( 2 ), 'o' );
%                         plot( position_rx( 1 ), position_rx( 2 ), 'o' );
%                         plot( position_FOV( 1 ), position_FOV( 2 ), 'o' );
%                         hold off;
%                         pause;

                    end % for index_active_rx = 1:index_active_rx_ub

                    % stop time measurement per tx
                    seconds_per_tx( index_active_tx ) = toc( time_tx_start );

                    % erase progress in percent
                    fprintf( repmat( '\b', [ 1, N_bytes ] ) );

                end % for index_active_tx = 1:numel( indices_active_tx{ index_setup } )

                % complete symmetric matrices
                if flag_symmetric
                    t_tof_lbs = t_tof_lbs + tril( t_tof_lbs, -1 ).';
                    t_tof_ubs = t_tof_ubs + tril( t_tof_ubs, -1 ).';
                end

                % create time intervals
                intervals_tof{ index_setup } = math.interval( t_tof_lbs, t_tof_ubs );

                % infer and print elapsed time
                time_elapsed = toc( time_start );
                fprintf( 'done! (%f s)\n', time_elapsed );

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
            % ensure class scattering.sequences.setups.setup
            if ~isa( setups, 'scattering.sequences.setups.setup')
                errorStruct.message = 'setups must be scattering.sequences.setups.setup!';
                errorStruct.identifier = 'discretize:NoSetups';
                error( errorStruct );
            end

            % ensure class scattering.sequences.setups.discretizations.options
            if ~isa( options_spatial, 'scattering.sequences.setups.discretizations.options')
                errorStruct.message = 'options_spatial must be scattering.sequences.setups.discretizations.options!';
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
                setups = scattering.sequences.setups.setup_grid_symmetric( [ setups.xdc_array ], [ setups.homogeneous_fluid ], [ setups.FOV ], [ setups.str_name ] );
            catch
                message = 'The discrete representation of the setup is asymmetric! This significantly increases the computational costs!';
                identifier = 'discretize:AsymmetricSetup';
                warning( identifier, message );
            end

        end % function setups = discretize( setups, options_spatial )

        %------------------------------------------------------------------
        % compute prefactors (- delta_V * axis_k_tilde.^2)
        %------------------------------------------------------------------
        function prefactors = compute_prefactors( setups, axes_f )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % ensure class scattering.sequences.setups.setup
            if ~isa( setups, 'scattering.sequences.setups.setup' )
                errorStruct.message = 'setups must be scattering.sequences.setups.setup!';
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
                if ~( isa( setups( index_object ).FOV.shape, 'scattering.sequences.setups.geometry.orthotope_grid' ) && isa( setups( index_object ).FOV.shape.grid, 'math.grid_regular' ) )
                    errorStruct.message = sprintf( 'setups( %d ).FOV.shape must be scattering.sequences.setups.geometry.orthotope_grid and setups( %d ).FOV.shape.grid must be math.grid_regular!', index_object, index_object );
                    errorStruct.identifier = 'compute_prefactors:NoRegularGridInDiscretizedOrthotope';
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
            prefactors = processing.signal_matrix( axes_f, samples );

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
            % ensure class scattering.sequences.setups.setup
            if ~isa( setups, 'scattering.sequences.setups.setup' )
                errorStruct.message = 'setups must be scattering.sequences.setups.setup!';
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
                        temp = processing.greens_function( face_act.shape.grid, setups( index_grid ).FOV.shape.grid, axis_k_tilde, indices{ index_batch } );

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
                h_transfer{ index_grid } = processing.field( repmat( axes_f( index_grid ), size( h_samples ) ), repmat( setups( index_grid ).FOV.shape.grid, size( h_samples ) ), h_samples );

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
        function h_transfer_aa = anti_aliasing_filter( setups, h_transfer, options_anti_aliasing, indices_element )
        % apply anti-aliasing filter to
        % the spatial transfer function for the d-dimensional Euclidean space

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % ensure class scattering.sequences.setups.setup
            if ~isa( setups, 'scattering.sequences.setups.setup' )
                errorStruct.message = 'setups must be scattering.sequences.setups.setup!';
                errorStruct.identifier = 'anti_aliasing_filter:NoSetups';
                error( errorStruct );
            end

            % ensure class scattering.sequences.setups.transducers.array_planar_regular_orthogonal
            indicator = cellfun( @( x ) ~isa( x, 'scattering.sequences.setups.transducers.array_planar_regular_orthogonal' ), { setups.xdc_array } );
            if any( indicator( : ) )
                errorStruct.message = 'setups.xdc_array must be scattering.sequences.setups.transducers.array_planar_regular_orthogonal!';
                errorStruct.identifier = 'anti_aliasing_filter:NoOrthogonalRegularPlanarArrays';
                error( errorStruct );
            end

            % ensure class processing.field
            if ~isa( h_transfer, 'processing.field' )
                errorStruct.message = 'h_transfer must be processing.field!';
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
            if nargin < 4 || isempty( indices_element )
                indices_element = num2cell( 1 );
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

                    % exclude dimensions with less than two array elements
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
            h_transfer_aa = processing.field( [ h_transfer.axis ], [ h_transfer.grid_FOV ], h_samples_aa );

        end % function h_transfer_aa = anti_aliasing_filter( setups, h_transfer, options_anti_aliasing, indices_element )

        %------------------------------------------------------------------
        % compute flag reflecting the local angular spatial frequencies (scalar)
        %------------------------------------------------------------------
        function flag = compute_flag( ~, setup, h_transfer, index_element )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % calling method ensures class scattering.options.anti_aliasing for filter (scalar)
            % calling method ensures class scattering.sequences.setups.setup for setup (scalar)
            % calling method ensures class scattering.sequences.setups.transducers.array_planar_regular_orthogonal for setup.xdc_array (scalar)
            % calling method ensures class processing.field for h_transfer (scalar)
            % calling method ensures ensure nonempty indices_element

            %--------------------------------------------------------------
            % 2.) apply spatial anti-aliasing filter (scalar)
            %--------------------------------------------------------------
            % numbers of discrete frequencies
            N_samples_f = abs( h_transfer.axis );

            % compute lateral components of mutual unit vectors
            e_1_minus_2 = mutual_unit_vectors( math.grid( setup.xdc_array.positions_ctr ), h_transfer.grid_FOV, index_element );
            e_1_minus_2 = repmat( abs( e_1_minus_2( :, :, 1:( end - 1 ) ) ), [ N_samples_f, 1 ] );

            % exclude dimensions with less than two array elements
            indicator_dimensions = setup.xdc_array.N_elements_axis > 1;
            N_dimensions_lateral_relevant = sum( indicator_dimensions );
            e_1_minus_2 = e_1_minus_2( :, :, indicator_dimensions );

            % compute flag reflecting the local angular spatial frequencies
            axis_k_tilde = compute_wavenumbers( setup.homogeneous_fluid.absorption_model, h_transfer.axis );
            flag = real( axis_k_tilde.members ) .* e_1_minus_2 .* reshape( setup.xdc_array.cell_ref.edge_lengths( indicator_dimensions ), [ 1, 1, N_dimensions_lateral_relevant ] );

        end % function flag = compute_flag( ~, setup, h_transfer, index_element )

        %------------------------------------------------------------------
        % is discretized setup symmetric
        %------------------------------------------------------------------
        function [ tf, N_points_per_pitch_axis ] = issymmetric( setups )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % ensure class scattering.sequences.setups.setup
            if ~isa( setups, 'scattering.sequences.setups.setup')
                errorStruct.message = 'setups must be scattering.sequences.setups.setup!';
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
                % ensure class scattering.sequences.setups.transducers.array_planar_regular_orthogonal
                if ~isa( setups( index_setup ).xdc_array, 'scattering.sequences.setups.transducers.array_planar_regular_orthogonal' )
                    errorStruct.message = sprintf( 'setups( %d ).xdc_array must be scattering.sequences.setups.transducers.array_planar_regular_orthogonal!', index_setup );
                    errorStruct.identifier = 'issymmetric:NoOrthogonalRegularPlanarArray';
                    error( errorStruct );
                end

                %----------------------------------------------------------
                % b) ensure orthotopes discretized by orthogonal regular grids
                %----------------------------------------------------------
                % method issymmetric of vibrating face ensures class scattering.sequences.setups.geometry.orthotope_grid
                % method issymmetric of vibrating face ensures class math.grid_regular_orthogonal

% TODO: move to fields_of_view.field_of_view
                % ensure class scattering.sequences.setups.geometry.orthotope_grid
                if ~isa( setups( index_setup ).FOV.shape, 'scattering.sequences.setups.geometry.orthotope_grid' )
                    errorStruct.message = sprintf( 'setups( %d ).FOV.shape must be scattering.sequences.setups.geometry.orthotope_grid!', index_setup );
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
                % ensure lateral symmetries of the vibrating faces
                indicator = issymmetric( setups( index_setup ).xdc_array.aperture );
                if any( ~indicator( : ) )
                    errorStruct.message = 'Symmetric spatial grid requires the lateral symmetries of the vibrating faces!';
                    errorStruct.identifier = 'issymmetric:NoSymmetry';
                    error( errorStruct );
                end

                % ensure lateral symmetries of the FOV about the axial axis
                FOV_pos_ctr = 2 * setups( index_setup ).FOV.shape.grid.offset_axis( 1:(end - 1) ) + ( setups( index_setup ).FOV.shape.grid.N_points_axis( 1:(end - 1) ) - 1 ) .* setups( index_setup ).FOV.shape.grid.cell_ref.edge_lengths( 1:(end - 1) );
                if any( abs( double( FOV_pos_ctr ) ) > eps( 0 ) )
                    errorStruct.message = 'Symmetric spatial grid requires the lateral symmetries of the FOV about the axial axis!';
                    errorStruct.identifier = 'issymmetric:NoSymmetry';
                    error( errorStruct );
                end

                %----------------------------------------------------------
                % d) lateral spacing is an integer fraction of the element pitch
                %    => translational invariance by shifts of factor_interp_tx points
                %----------------------------------------------------------
                N_points_per_pitch_axis{ index_setup } = setups( index_setup ).xdc_array.cell_ref.edge_lengths ./ setups( index_setup ).FOV.shape.grid.cell_ref.edge_lengths( 1:( end - 1 ) );
                if any( abs( N_points_per_pitch_axis{ index_setup } - round( N_points_per_pitch_axis{ index_setup } ) ) > eps( round( N_points_per_pitch_axis{ index_setup } ) ) )
                    errorStruct.message = 'Symmetric discretized setup requires the lateral spacings of the grid points in the FOV to be integer fractions of the element pitch!';
                    errorStruct.identifier = 'issymmetric:NoIntegerFraction';
                    error( errorStruct );
                end

                %----------------------------------------------------------
                % e) TODO: check minimal # of lateral grid points
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

                % declare discretized setup to be symmetric
                tf( index_setup ) = true;
                N_points_per_pitch_axis{ index_setup } = round( N_points_per_pitch_axis{ index_setup } );

            end % for index_setup = 1:numel( setups )

            % avoid cell array for single setups
            if isscalar( setups )
                N_points_per_pitch_axis = N_points_per_pitch_axis{ 1 };
            end

        end % function [ tf, N_points_per_pitch_axis ] = issymmetric( setups )

        %
        function setup_grid_symmetric( setups )
        end

    end % methods

end % classdef setup
