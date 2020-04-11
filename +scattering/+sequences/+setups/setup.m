%
% superclass for all pulse-echo measurement setups
%
% author: Martin F. Schiffner
% date: 2018-03-12
% modified: 2020-04-09
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
        function intervals_tof = times_of_flight( setups, indices_active_tx, indices_active_rx )

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
            if nargin < 2 || isempty( indices_active_tx )
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
            if nargin < 3 || isempty( indices_active_rx )
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
        % adjust recording time intervals
        %------------------------------------------------------------------
        function settings_rx = adjust_intervals_t( setup, settings_tx, settings_rx )
% TODO: vectorize for multiple setups!
            % print status
            time_start = tic;
            str_date_time = sprintf( '%04d-%02d-%02d: %02d:%02d:%02d', fix( clock ) );
            fprintf( '\t %s: adjusting recording time intervals...', str_date_time );

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % ensure class scattering.sequences.setups.setup (scalar)
            if ~( isa( setup, 'scattering.sequences.setups.setup' ) && isscalar( setup ) )
                errorStruct.message = 'setup must be a single scattering.sequences.setups.setup!';
                errorStruct.identifier = 'set_intervals_t:NoSetup';
                error( errorStruct );
            end

            % ensure class scattering.sequences.settings.controls.tx
            if ~isa( settings_tx, 'scattering.sequences.settings.controls.tx' )
                errorStruct.message = 'settings_tx must be scattering.sequences.settings.controls.tx!';
                errorStruct.identifier = 'set_intervals_t:NoSettingsTx';
                error( errorStruct );
            end

            % ensure cell array for settings_rx
            if ~iscell( settings_rx )
                settings_rx = { settings_rx };
            end

            % ensure equal number of dimensions and sizes
            [ settings_tx, settings_rx ] = auxiliary.ensureEqualSize( settings_tx, settings_rx );

            %--------------------------------------------------------------
            % 2.) recording time intervals
            %--------------------------------------------------------------
            % specify cell array for intervals_t
            intervals_t = cell( size( settings_tx ) );

            % iterate synthesis settings
            for index_tx = 1:numel( settings_tx )

                % ensure class scattering.sequences.settings.controls.rx
                if ~isa( settings_rx{ index_tx }, 'scattering.sequences.settings.controls.rx' )
                    errorStruct.message = 'settings_rx must be scattering.sequences.settings.controls.rx!';
                    errorStruct.identifier = 'set_intervals_t:NoSettingsRx';
                    error( errorStruct );
                end

                % number of active tx elements
                N_active_tx = numel( settings_tx( index_tx ).indices_active );

                % initialize lower and upper bounds on the support w/ zeros
                t_lbs = physical_values.second( zeros( size( settings_rx{ index_tx } ) ) );
                t_ubs = physical_values.second( zeros( size( settings_rx{ index_tx } ) ) );

                % iterate mixed voltage signals
                for index_rx = 1:numel( settings_rx{ index_tx } )

                    % number of active rx elements
                    N_active_rx = numel( settings_rx{ index_tx }( index_rx ).indices_active );

                    % initialize lower and upper bounds on the support
                    t_lbs_all = physical_values.second( zeros( N_active_tx, N_active_rx ) );
                    t_ubs_all = physical_values.second( zeros( N_active_tx, N_active_rx ) );

                    % iterate active rx elements
                    for index_active_rx = 1:N_active_rx

                        % active rx element
                        index_element_rx = settings_rx{ index_tx }( index_rx ).indices_active( index_active_rx );

                        % support of impulse responses
                        t_lb_rx_act = settings_rx{ index_tx }( index_rx ).impulse_responses.axis.members( 1 );
                        t_ub_rx_act = settings_rx{ index_tx }( index_rx ).impulse_responses.axis.members( end );

                        % iterate active tx elements
                        for index_active_tx = 1:N_active_tx

                            % active tx element
                            index_element_tx = settings_tx( index_tx ).indices_active( index_active_tx );

                            % support of excitation_voltages
                            indicator = double( abs( settings_tx( index_tx ).excitation_voltages.samples( :, index_active_tx ) ) ) >= eps;
                            members = settings_tx( index_tx ).excitation_voltages.axis.members( indicator );
                            t_lb_tx_act = members( 1 );
                            t_ub_tx_act = members( end );

                            % support of impulse responses
                            indicator = double( abs( settings_tx( index_tx ).impulse_responses.samples( :, index_active_tx ) ) ) >= eps;
                            members = settings_tx( index_tx ).impulse_responses.axis.members( indicator );
                            t_lb_tx_act = t_lb_tx_act + members( 1 );
                            t_ub_tx_act = t_ub_tx_act + members( end );

                            % compute lower and upper bounds on the recording time intervals
                            t_lbs_all( index_active_tx, index_active_rx ) = t_lb_rx_act + t_lb_tx_act + setup.intervals_tof( index_element_tx, index_element_rx ).lb;
                            t_ubs_all( index_active_tx, index_active_rx ) = t_ub_rx_act + t_ub_tx_act + setup.intervals_tof( index_element_tx, index_element_rx ).ub;

                        end % for index_active_tx = 1:N_active_tx

                    end % for index_active_rx = 1:N_active_rx

                    % extract maximum support
                    t_lbs( index_rx ) = min( t_lbs_all( : ) );
                    t_ubs( index_rx ) = max( t_ubs_all( : ) );

                end % for index_rx = 1:numel( settings_rx{ index_tx } )

                % create time intervals for all mixes
                intervals_t{ index_tx } = math.interval( t_lbs, t_ubs );

                % intersect with recording time intervals
                settings_rx{ index_tx } = intersect( settings_rx{ index_tx }, intervals_t{ index_tx } );

            end % for index_tx = 1:numel( settings_tx )

            % avoid cell array for single settings_tx
            if isscalar( settings_tx )
                settings_rx = settings_rx{ 1 };
            end

            % infer and print elapsed time
            time_elapsed = toc( time_start );
            fprintf( 'done! (%f s)\n', time_elapsed );

        end % function settings_rx = adjust_intervals_t( setup, settings_tx, settings_rx )

        %------------------------------------------------------------------
        % compute excitation voltages
        %------------------------------------------------------------------
        function [ excitation_voltages, indices_active ] = compute_excitation_voltages( setups, u_tx_tilde, waves )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % ensure class scattering.sequences.setups.setup
            if ~isa( setups, 'scattering.sequences.setups.setup' )
                errorStruct.message = 'setups must be scattering.sequences.setups.setup!';
                errorStruct.identifier = 'compute_excitation_voltages:NoSetups';
                error( errorStruct );
            end

            % ensure cell array for u_tx_tilde
            if ~iscell( u_tx_tilde )
                u_tx_tilde = { u_tx_tilde };
            end

            % ensure cell array for waves
            if ~iscell( waves )
                waves = { waves };
            end

            % ensure equal number of dimensions and sizes
            [ setups, u_tx_tilde, waves ] = auxiliary.ensureEqualSize( setups, u_tx_tilde, waves );

            %--------------------------------------------------------------
            % 2.) compute excitation voltages
            %--------------------------------------------------------------
            % specify cell arrays
            excitation_voltages = cell( size( setups ) );
            indices_active = cell( size( setups ) );

            % iterate pulse-echo measurement setups
            for index_setup = 1:numel( setups )

                %----------------------------------------------------------
                % a) check arguments
                %----------------------------------------------------------
                % ensure class processing.signal_matrix
                if ~isa( u_tx_tilde{ index_setup }, 'processing.signal_matrix' )
                    errorStruct.message = 'u_tx_tilde must be processing.signal_matrix!';
                    errorStruct.identifier = 'compute_excitation_voltages:NoSignalArrays';
                    error( errorStruct );
                end
% TODO: regular axis!!!
                % ensure class scattering.sequences.syntheses.wave
                if ~isa( waves{ index_setup }, 'scattering.sequences.syntheses.wave' )
                    errorStruct.message = 'waves must be scattering.sequences.syntheses.wave!';
                    errorStruct.identifier = 'compute_excitation_voltages:NoWaves';
                    error( errorStruct );
                end

                % ensure equal number of dimensions and sizes
                [ u_tx_tilde{ index_setup }, waves{ index_setup } ] = auxiliary.ensureEqualSize( u_tx_tilde{ index_setup }, waves{ index_setup } );

                %----------------------------------------------------------
                % b) compute excitation voltages
                %----------------------------------------------------------
                % compute time delays and apodization weights
                [ time_delays, apodization_weights, indices_active{ index_setup } ] = compute_delays( waves{ index_setup }, setups( index_setup ).xdc_array, setups( index_setup ).homogeneous_fluid.c_avg );

                % ensure cell arrays
                if ~iscell( time_delays )
                    time_delays = { time_delays };
                    apodization_weights = { apodization_weights };
                end

                % specify cell array for excitation_voltages
                excitation_voltages{ index_setup } = cell( size( u_tx_tilde{ index_setup } ) );

                % iterate incident waves
                for index_wave = 1:numel( u_tx_tilde{ index_setup } )

                    % quantize time delays
                    indices_q = round( time_delays{ index_wave } / setups( index_setup ).T_clk );

                    %------------------------------------------------------
                    % b) apply time delays to reference voltage signals
                    %------------------------------------------------------
% TODO: compute length differently!
                    % unique deltas
                    deltas_unique = [ u_tx_tilde{ index_setup }( index_wave ).axis.delta, setups( index_setup ).T_clk ];

                    % largest delta_unique must be integer multiple of smaller deltas_unique
                    delta_unique_max = max( deltas_unique );
                    factors_int = round( delta_unique_max ./ deltas_unique );
                    if any( abs( delta_unique_max ./ deltas_unique - factors_int ) > eps( factors_int ) )
                        errorStruct.message = 'delta_unique_max must be integer multiple of all deltas_unique!';
                        errorStruct.identifier = 'compute_excitation_voltages:NoIntegerMultiple';
                        error( errorStruct );
                    end

                    % quantize new time duration using largest delta
                    T_ref = ceil( ( abs( u_tx_tilde{ index_setup }( index_wave ).axis ) * u_tx_tilde{ index_setup }( index_wave ).axis.delta + max( indices_q ) * setups( index_setup ).T_clk ) / delta_unique_max ) * delta_unique_max;

                    % compute Fourier representations
                    u_tx = fourier_coefficients( u_tx_tilde{ index_setup }( index_wave ), T_ref );
                    impulse_responses = processing.signal_matrix( u_tx.axis, apodization_weights{ index_wave }( : ).' .* exp( -2j * pi * u_tx.axis.members * setups( index_setup ).T_clk * indices_q( : ).' ) );
                    excitation_voltages{ index_setup }{ index_wave } = signal( impulse_responses .* u_tx, double( u_tx_tilde{ index_setup }( index_wave ).axis.q_lb ), u_tx_tilde{ index_setup }( index_wave ).axis.delta );

                end % for index_wave = 1:numel( u_tx_tilde{ index_setup } )

                % reshape excitation_voltages
                excitation_voltages{ index_setup } = reshape( cat( 1, excitation_voltages{ index_setup }{ : } ), size( waves{ index_setup } ) );

            end % for index_setup = 1:numel( setups )

            % avoid cell arrays for single setups
            if isscalar( setups )
                excitation_voltages = excitation_voltages{ 1 };
                indices_active = indices_active{ 1 };
            end

        end % function [ excitation_voltages, indices_active ] = compute_excitation_voltages( setups, u_tx_tilde, waves )

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
        % compute spatial transfer functions
        %------------------------------------------------------------------
        function h_transfer = transfer_function( setups, axes_f, indices_element, filters )
% TODO: add gradient?
            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % ensure class scattering.sequences.setups.setup
            if ~isa( setups, 'scattering.sequences.setups.setup' )
                errorStruct.message = 'setups must be scattering.sequences.setups.setup!';
                errorStruct.identifier = 'transfer_function:NoSetups';
                error( errorStruct );
            end

% TODO: ensure discretized setup -> faces with regular grids
% cellfun( @( x ) x.aperture, { setups.xdc_array } )

% TODO: ensure class math.grid_regular
%             if ~isa( face_act.shape.grid, 'math.grid_regular' )
%                 errorStruct.message = 'face_act.shape.grid must be math.grid_regular!';
%                 errorStruct.identifier = 'transfer_function_scalar:NoRegularGrid';
%                 error( errorStruct );
%             end
            
            % ensure class math.sequence_increasing with physical_values.frequency members
            if ~( isa( axes_f, 'math.sequence_increasing' ) && all( cellfun( @( x ) isa( x, 'physical_values.frequency' ), { axes_f.members } ) ) )
                errorStruct.message = 'axes_f must be math.sequence_increasing with physical_values.frequency members!';
                errorStruct.identifier = 'transfer_function:InvalidFrequencyAxes';
                error( errorStruct );
            end

            % ensure nonempty indices_element
            if nargin < 3 || isempty( indices_element )
                indices_element = 1;
            end

            % ensure cell array for indices_element
            if ~iscell( indices_element )
                indices_element = { indices_element };
            end

            % ensure nonempty filters
            if nargin < 4 || isempty( filters )
                filters = scattering.anti_aliasing_filters.off;
            end

            % ensure class scattering.anti_aliasing_filters.anti_aliasing_filter
            if ~isa( filters, 'scattering.anti_aliasing_filters.anti_aliasing_filter' )
                errorStruct.message = 'filters must be scattering.anti_aliasing_filters.anti_aliasing_filter!';
                errorStruct.identifier = 'transfer_function:NoSpatialAntiAliasingFilters';
                error( errorStruct );
            end

            % multiple setups / single axes_f
            if ~isscalar( setups ) && isscalar( axes_f )
                axes_f = repmat( axes_f, size( setups ) );
            end

            % multiple setups / single indices_element
            if ~isscalar( setups ) && isscalar( indices_element )
                indices_element = repmat( indices_element, size( setups ) );
            end

            % multiple setups / single filters
            if ~isscalar( setups ) && isscalar( filters )
                filters = repmat( filters, size( setups ) );
            end

            % ensure equal number of dimensions and sizes
            auxiliary.mustBeEqualSize( setups, axes_f, indices_element, filters );

            %--------------------------------------------------------------
            % 2.) compute spatial transfer functions
            %--------------------------------------------------------------
            % specify cell array for h_transfer
            h_transfer = cell( size( setups ) );

            % iterate discretized pulse-echo measurement setups
            for index_setup = 1:numel( setups )

                %----------------------------------------------------------
                % a) validate indices of selected array elements
                %----------------------------------------------------------
                % ensure nonempty positive integers
                mustBeNonempty( indices_element{ index_setup } );
                mustBeInteger( indices_element{ index_setup } );
                mustBePositive( indices_element{ index_setup } );

                % ensure that indices_element{ index_setup } does not exceed the number of array elements
                if any( indices_element{ index_setup }( : ) > setups( index_setup ).xdc_array.N_elements )
                    errorStruct.message = sprintf( 'indices_element{ %d } must not exceed the number of array elements %d!', index_setup, setups( index_setup ).xdc_array.N_elements );
                    errorStruct.identifier = 'transfer_function:InvalidElementIndices';
                    error( errorStruct );
                end

                %----------------------------------------------------------
                % b) compute spatial transfer functions for selected array elements
                %----------------------------------------------------------
                % specify cell array for h_samples
                h_samples = cell( size( indices_element{ index_setup } ) );

                % iterate selected array elements
                for index_selected = 1:numel( indices_element{ index_setup } )

                    % index of current array element
                    index_element = indices_element{ index_setup }( index_selected );

                    % create format string for filename
                    str_format = sprintf( 'data/%s/setup_%%s/h_%d_axis_f_%%s.mat', setups( index_setup ).str_name, index_element );

                    % load or compute spatial transfer function
% TODO: loading and saving optional
                    h_samples{ index_selected } ...
                    = auxiliary.compute_or_load_hash( str_format, @transfer_function_scalar, [ 4, 2 ], [ 1, 2, 3 ], ...
                      setups( index_setup ), axes_f( index_setup ), index_element, ...
                      { setups( index_setup ).xdc_array.aperture, setups( index_setup ).homogeneous_fluid, setups( index_setup ).FOV, setups( index_setup ).str_name } );

                end % for index_selected = 1:numel( indices_element{ index_setup } )

                %----------------------------------------------------------
                % c) create fields
                %----------------------------------------------------------
                h_transfer{ index_setup } = processing.field( axes_f( index_setup ), setups( index_setup ).FOV.shape.grid, h_samples );

                %----------------------------------------------------------
                % d) apply spatial anti-aliasing filter
                %----------------------------------------------------------
% TODO: multiple indices_element{ index_setup } valid?
                h_transfer{ index_setup } = apply( filters( index_setup ), setups( index_setup ), h_transfer{ index_setup }, indices_element{ index_setup } );

            end % for index_setup = 1:numel( setups )

            % avoid cell array for single setups
            if isscalar( setups )
                h_transfer = h_transfer{ 1 };
            end

        end % function h_transfer = transfer_function( setups, axes_f, indices_element, filters )

        %------------------------------------------------------------------
        % compute flags reflecting the local angular spatial frequencies
        %------------------------------------------------------------------
        function flags = compute_flags( setups, axes_f, indices_element )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % ensure class scattering.sequences.setups.setup
            if ~isa( setups, 'scattering.sequences.setups.setup' )
                errorStruct.message = 'setups must be scattering.sequences.setups.setup!';
                errorStruct.identifier = 'compute_flags:NoSetups';
                error( errorStruct );
            end

            % ensure class math.sequence_increasing with physical_values.frequency members
            if ~( isa( axes_f, 'math.sequence_increasing' ) && all( cellfun( @( x ) isa( x, 'physical_values.frequency' ), { axes_f.members } ) ) )
                errorStruct.message = 'axes_f must be math.sequence_increasing with physical_values.frequency members!';
                errorStruct.identifier = 'compute_flags:InvalidFrequencyAxes';
                error( errorStruct );
            end

            % ensure nonempty indices_element
            if nargin < 3 || isempty( indices_element )
                indices_element = 1;
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
            % 2.) compute flags reflecting the local angular spatial frequencies
            %--------------------------------------------------------------
            % numbers of discrete frequencies
            N_samples_f = abs( axes_f );

            % specify cell array for flags
            flags = cell( size( setups ) );

            % iterate pulse-echo measurement setups
            for index_setup = 1:numel( setups )

                %----------------------------------------------------------
                % a) validate indices of selected array elements
                %----------------------------------------------------------
                % ensure nonempty positive integers
                mustBeInteger( indices_element{ index_setup } );
                mustBePositive( indices_element{ index_setup } );
                mustBeNonempty( indices_element{ index_setup } );

                % ensure that indices_element{ index_setup } does not exceed the number of array elements
                if any( indices_element{ index_setup }( : ) > setups( index_setup ).xdc_array.N_elements )
                    errorStruct.message = sprintf( 'indices_element{ %d } must not exceed the number of array elements %d!', index_setup, setups( index_setup ).xdc_array.N_elements );
                    errorStruct.identifier = 'compute_flags:InvalidElementIndices';
                    error( errorStruct );
                end

                %----------------------------------------------------------
                % b) compute flags for selected array elements
                %----------------------------------------------------------
                % compute current complex-valued wavenumbers
                axis_k_tilde = compute_wavenumbers( setups( index_setup ).homogeneous_fluid.absorption_model, axes_f( index_setup ) );

                % compute absolute lateral components of mutual unit vectors
                e_1_minus_2 = mutual_unit_vectors( math.grid( setups( index_setup ).xdc_array.positions_ctr ), setups( index_setup ).FOV.shape.grid, indices_element{ index_setup } );
                e_1_minus_2 = abs( e_1_minus_2( :, :, 1:( end - 1 ) ) );

                % exclude lateral dimensions with less than two array elements
                indicator_dimensions = setups( index_setup ).xdc_array.N_elements_axis > 1;
                N_dimensions_lateral_relevant = sum( indicator_dimensions );
                e_1_minus_2 = e_1_minus_2( :, :, indicator_dimensions );

                % local phase shift per element pitch (rad)
                beta_times_delta = real( axis_k_tilde.members ) .* reshape( setups( index_setup ).xdc_array.cell_ref.edge_lengths( indicator_dimensions ), [ 1, 1, N_dimensions_lateral_relevant ] );

                % specify cell array for flags{ index_setup }
                flags{ index_setup } = cell( size( indices_element{ index_setup } ) );

                % iterate selected array elements
                for index_selected = 1:numel( indices_element{ index_setup } )

                    % select absolute lateral components of mutual unit vectors
                    e_1_minus_2_act = repmat( e_1_minus_2( index_selected, :, : ), [ N_samples_f( index_setup ), 1, 1 ] );

                    % lateral local phase shift per element pitch (rad)
                    flags{ index_setup }{ index_selected } = beta_times_delta .* e_1_minus_2_act;

                end % for index_selected = 1:numel( indices_element{ index_setup } )

                % create field arrays
% TODO: problem with three-dimensional flags{ index_setup }{ index_selected }!
                flags{ index_setup } = processing.field( axes_f( index_setup ), setups( index_setup ).FOV.shape.grid, flags{ index_setup } );

            end % for index_setup = 1:numel( setups )

            % avoid cell array for single setups
            if isscalar( setups )
                flags = flags{ 1 };
            end

        end % function flags = compute_flags( setups, axes_f, indices_element )

        %------------------------------------------------------------------
        % compute incident acoustic pressure fields
        %------------------------------------------------------------------
        function fields = compute_p_in( setups, settings_tx, filters )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % ensure class scattering.sequences.setups.setup
            if ~isa( setups, 'scattering.sequences.setups.setup' )
                errorStruct.message = 'setups must be scattering.sequences.setups.setup!';
                errorStruct.identifier = 'compute_p_in:NoSetups';
                error( errorStruct );
            end

            % ensure cell array for settings_tx
            if ~iscell( settings_tx )
                settings_tx = { settings_tx };
            end

            % ensure equal number of dimensions and sizes
            auxiliary.mustBeEqualSize( setups, settings_tx, filters );

            %--------------------------------------------------------------
            % 2.) compute incident acoustic pressure fields
            %--------------------------------------------------------------
            % specify cell arrays
            fields = cell( size( setups ) );

            % iterate setups of pulse-echo measurements
            for index_setup = 1:numel( setups )

                % iterate
                index_tx = 1;
                fields{ index_setup } = compute_p_in_scalar( setups( index_setup ), settings_tx{ index_setup }( index_tx ).tx.indices_active, settings_tx{ index_setup }( index_tx ).v_d_unique, filters( index_setup ) );

            end % for index_setup = 1:numel( setups )

            fields = processing.field( axes_f_measurement_unique, setups( index_setup ).FOV.shape.grid, fields{ index_sequence } );

        end % function fields = compute_p_in( setups, settings_tx, filters )

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

	end % methods

	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	%% methods (private and hidden)
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	methods (Access = private, Hidden)

        %------------------------------------------------------------------
        % compute excitation voltages (scalar)
        %------------------------------------------------------------------

        %------------------------------------------------------------------
        % compute spatial transfer function (scalar)
        %------------------------------------------------------------------
        function h_samples = transfer_function_scalar( setup, axis_f, index_element )

            % internal constant
            N_points_max = 1;

            % print status
            time_start = tic;
            str_date_time = sprintf( '%04d-%02d-%02d: %02d:%02d:%02d', fix( clock ) );
            fprintf( '\t %s: computing spatial transfer function... ', str_date_time );

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % calling function ensures class scattering.sequences.setups.setup (scalar) for setup
            % calling function ensures class math.grid_regular for setup.xdc_array.aperture.shape.grid
            % calling function ensures class math.sequence_increasing with physical_values.frequency members (scalar) for axis_f
            % calling function ensures nonempty positive integer that does not exceed the number of array elements for index_element

            %--------------------------------------------------------------
            % 2.) compute spatial transfer function (scalar)
            %--------------------------------------------------------------
            % extract discretized vibrating face
            face_act = setup.xdc_array.aperture( index_element );

            % compute complex-valued apodization weights
            weights = compute_weights( face_act, axis_f );

            % number of discrete frequencies
            N_samples_f = abs( axis_f );

            % compute current complex-valued wavenumbers
            axis_k_tilde = compute_wavenumbers( setup.homogeneous_fluid.absorption_model, axis_f );

            % initialize samples with zeros
            h_samples = physical_values.meter( zeros( N_samples_f, setup.FOV.shape.grid.N_points ) );

            % partition grid points on vibrating face into batches to save memory
            N_batches = ceil( face_act.shape.grid.N_points / N_points_max );
            N_points_last = face_act.shape.grid.N_points - ( N_batches - 1 ) * N_points_max;
            indices = mat2cell( (1:face_act.shape.grid.N_points), 1, [ N_points_max * ones( 1, N_batches - 1 ), N_points_last ] );

            % iterate batches
            for index_batch = 1:N_batches

                % print progress in percent
                fprintf( '%5.1f %%', ( index_batch - 1 ) / N_batches * 1e2 );

                % compute Green's functions for specified pairs of grids and specified grid points
                temp = processing.greens_function( face_act.shape.grid, setup.FOV.shape.grid, axis_k_tilde, indices{ index_batch } );

                % apply complex-valued apodization weights
                temp = weights( indices{ index_batch }, :, : ) .* temp;

                % integrate over aperture
                h_samples = h_samples - 2 * face_act.shape.grid.cell_ref.volume * shiftdim( sum( temp, 1 ), 1 ).';

                % erase progress in percent
                fprintf( '\b\b\b\b\b\b\b' );

            end % for index_batch = 1:N_batches

            % infer and print elapsed time
            time_elapsed = toc( time_start );
            fprintf( 'done! (%f s)\n', time_elapsed );

        end % function h_samples = transfer_function_scalar( setup, axis_f, index_element )

        %------------------------------------------------------------------
        % compute incident acoustic pressure field (scalar)
        %------------------------------------------------------------------
        function p_in_samples = compute_p_in_scalar( setup, indices_active, v_d, filter )

            % print status
            time_start = tic;
            str_date_time = sprintf( '%04d-%02d-%02d: %02d:%02d:%02d', fix( clock ) );
            fprintf( '\t %s: computing incident acoustic pressure field (kappa)...', str_date_time );

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % calling function ensures class scattering.sequences.setups.setup (scalar) for setup
            % calling function ensures nonempty positive integers that do not exceed the number of array elements for indices_active
            % calling function ensures class processing.signal_matrix (scalar) for v_d
            % calling function ensures class scattering.anti_aliasing_filters.anti_aliasing_filter (scalar) for filter

            %--------------------------------------------------------------
            % 2.) compute incident acoustic pressure field (scalar)
            %--------------------------------------------------------------
            % extract frequency axis
            axis_f = v_d.axis;
            N_samples_f = abs( axis_f );

            % initialize pressure samples with zeros
            p_in_samples = physical_values.pascal( zeros( N_samples_f, setup.FOV.shape.grid.N_points ) );

            % iterate active array elements
            for index_active = 1:numel( indices_active )

                % index of active array element
                index_element = indices_active( index_active );

                % compute spatial transfer function of the active array element
                h_tx_unique = transfer_function( setup, axis_f, index_element, filter );
                h_tx_unique = double( h_tx_unique.samples );

                % compute summand for the incident pressure field
                p_in_samples_summand = h_tx_unique .* double( v_d.samples( :, index_active ) );

                % add summand to the incident pressure field
% TODO: correct unit problem
                p_in_samples = p_in_samples + physical_values.pascal( p_in_samples_summand );

            end % for index_active = 1:numel( indices_active )

            % infer and print elapsed time
            time_elapsed = toc( time_start );
            fprintf( 'done! (%f s)\n', time_elapsed );

        end % function p_in_samples = compute_p_in_scalar( setup, indices_active, v_d, filter )

	end % methods (Access = private, Hidden)

end % classdef setup
