%
% superclass for all pulse-echo measurement setups
%
% author: Martin F. Schiffner
% date: 2018-03-12
% modified: 2019-05-22
%
classdef setup

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% properties
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	properties (SetAccess = private)

        % independent properties
        xdc_array ( 1, 1 ) transducers.array = transducers.L14_5_38     % transducer array
        FOV ( 1, 1 ) fields_of_view.field_of_view                       % field of view
        % TODO: properties of the lossy homogeneous fluid
        absorption_model ( 1, 1 ) absorption_models.absorption_model = absorption_models.time_causal( 0, 0.5, 1, physical_values.meter_per_second( 1540 ), physical_values.hertz( 4e6 ), 1 )	% absorption model for the lossy homogeneous fluid
% TODO: c_avg vs c_ref?
% TODO: homogeneous_fluid
        c_avg = physical_values.meter_per_second( 1500 );               % average small-signal sound speed
        T_clk = physical_values.second( 1 / 80e6 );                     % time period of the clock signal
        str_name = 'default'                                            % name

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
        function objects = setup( xdc_arrays, FOVs, absorption_models, strs_name )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % ensure cell array for strs_name
            if ~iscell( strs_name )
                strs_name = { strs_name };
            end

            % ensure equal number of dimensions and sizes
            auxiliary.mustBeEqualSize( xdc_arrays, FOVs, absorption_models, strs_name );

            %--------------------------------------------------------------
            % 2.) create pulse-echo measurement setups
            %--------------------------------------------------------------
            % repeat default pulse-echo measurement setup
            objects = repmat( objects, size( xdc_arrays ) );

            % iterate pulse-echo measurement setups
            for index_object = 1:numel( objects )

                % ensure matching number of dimensions
                if xdc_arrays( index_object ).N_dimensions ~= ( FOVs( index_object ).N_dimensions - 1 )
                    errorStruct.message = sprintf( 'The number of dimensions in FOVs( %d ) must exceed that in xdc_arrays( %d ) by unity!', index_object, index_object );
                    errorStruct.identifier = 'setup:DimensionMismatch';
                    error( errorStruct );
                end

                %----------------------------------------------------------
                % a) set independent properties
                %----------------------------------------------------------
                objects( index_object ).xdc_array = xdc_arrays( index_object );
                objects( index_object ).FOV = FOVs( index_object );
                objects( index_object ).absorption_model = absorption_models( index_object );
                objects( index_object ).str_name = strs_name{ index_object };

                %----------------------------------------------------------
                % b) set dependent properties
                %----------------------------------------------------------
                objects( index_object ).intervals_tof = times_of_flight( objects( index_object ) );

            end % for index_object = 1:numel( objects )

        end % function objects = setup( xdc_arrays, FOVs, absorption_models, strs_name )

        %------------------------------------------------------------------
        % lower and upper bounds on the times-of-flight
        %------------------------------------------------------------------
        function intervals_tof = times_of_flight( setups, varargin )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % ensure planar transducer array and FOV with orthotope shape
            if ~( isa( [ setups.xdc_array ], 'transducers.array_planar' ) && isa( [ setups.FOV ], 'fields_of_view.orthotope' ) )
                errorStruct.message = 'Current implementation requires planar transducer array and FOV with orthotope shape!';
                errorStruct.identifier = 'times_of_flight:NoPlanarOrOrthotope';
                error( errorStruct );
            end

            % ensure nonempty indices_active_tx
            if nargin >= 2 && ~isempty( varargin{ 1 } )
                indices_active_tx = varargin{ 1 };
            else
                indices_active_tx = cell( size( setups ) );
                for index_object = 1:numel( setups )
                    indices_active_tx{ index_object } = ( 1:setups( index_object ).xdc_array.N_elements );
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
                for index_object = 1:numel( setups )
                    indices_active_rx{ index_object } = ( 1:setups( index_object ).xdc_array.N_elements );
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
            pos_vertices = vertices( [ setups.FOV ] );

            % ensure cell array for pos_vertices
            if ~iscell( pos_vertices )
                pos_vertices = { pos_vertices };
            end

            % specify cell array for intervals_tof
            intervals_tof = cell( size( setups ) );

            % iterate pulse-echo measurement setups
            for index_object = 1:numel( setups )

                % initialize lower and upper bounds with zeros
                t_tof_lbs = physical_values.second( zeros( numel( indices_active_tx{ index_object } ), numel( indices_active_rx{ index_object } ) ) );
                t_tof_ubs = physical_values.second( zeros( numel( indices_active_tx{ index_object } ), numel( indices_active_rx{ index_object } ) ) );

                % iterate active tx elements
                for index_active_tx = 1:numel( indices_active_tx{ index_object } )

                    % index of active tx element
                    index_element_tx = indices_active_tx{ index_object }( index_active_tx );

                    % planar face of active tx element
                    face_tx = setups( index_object ).xdc_array.aperture( index_element_tx );

                    % lower and upper bounds on the intervals
                    face_tx_lbs = [ face_tx.intervals.lb ];
                    face_tx_ubs = [ face_tx.intervals.ub ];

                    % iterate active rx elements
                    for index_active_rx = 1:numel( indices_active_rx{ index_object } )

                        % index of active rx element
                        index_element_rx = indices_active_rx{ index_object }( index_active_rx );

                        % planar face of active rx element
                        face_rx = setups( index_object ).xdc_array.aperture( index_element_rx );

                        % lower and upper bounds on the intervals
                        face_rx_lbs = [ face_rx.intervals.lb ];
                        face_rx_ubs = [ face_rx.intervals.ub ];

                        % orthotope including center coordinates of prolate spheroid
                        face_ctr_lbs = ( face_tx_lbs + face_rx_lbs ) / 2;
                        face_ctr_ubs = ( face_tx_ubs + face_rx_ubs ) / 2;
                        face_ctr_intervals = num2cell( math.interval( face_ctr_lbs, face_ctr_ubs ) );
                        face_ctr = math.orthotope( face_ctr_intervals{ : } );

                        % does lateral extent of FOV contain face_ctr?
%                         if intersection( setups( index_object ).FOV.intervals, face_ctr_intervals )

                            % distance from center coordinates to focal points
                            dist_focus_ctr = norm( face_rx.pos_center - face_tx.pos_center ) / 2;

                            %----------------------------------------------
                            % a) lower bound on the time-of-flight
                            %----------------------------------------------
                            t_tof_lbs( index_active_tx, index_active_rx ) = 2 * sqrt( dist_focus_ctr^2 + setups( index_object ).FOV.intervals( end ).lb^2 ) / setups( index_object ).c_avg;
                            t_tof_lbs( index_active_rx, index_active_tx ) = t_tof_lbs( index_active_tx, index_active_rx );

                            %----------------------------------------------
                            % b) upper bound on the time-of-flight
                            %----------------------------------------------
                            % determine vertices of maximum distance for lower and upper interval bounds
                            [ dist_ctr_vertices_max_lb, index_max_lb ] = max( vecnorm( [ [ face_ctr.intervals.lb ], 0 ] - pos_vertices{ index_object }, 2, 2 ) );
                            [ dist_ctr_vertices_max_ub, index_max_ub ] = max( vecnorm( [ [ face_ctr.intervals.ub ], 0 ] - pos_vertices{ index_object }, 2, 2 ) );

                            % find index and maximum distance
                            if dist_ctr_vertices_max_lb > dist_ctr_vertices_max_ub
                                index_max = index_max_lb;
                                % TODO:compute correct position
                                pos_tx = [ face_tx_lbs, 0 ];
                                pos_rx = [ face_rx_lbs, 0 ];
                            else
                                index_max = index_max_ub;
                                % TODO:compute correct position
                                pos_tx = [ face_tx_ubs, 0 ];
                                pos_rx = [ face_rx_ubs, 0 ];
                            end

                            % compute upper bound
                            t_tof_ubs( index_active_tx, index_active_rx ) = ( norm( pos_vertices{ index_object }( index_max, : ) - pos_tx ) + norm( pos_rx - pos_vertices{ index_object }( index_max, : ) ) ) / setups( index_object ).c_avg;
                            t_tof_ubs( index_active_rx, index_active_tx ) = t_tof_ubs( index_active_tx, index_active_rx );

%                         else
                            % find vertex intersecting with smallest prolate spheroid
%                         end

                    end % for index_active_rx = 1:numel( indices_active_rx{ index_object } )

                end % for index_active_tx = 1:numel( indices_active_tx{ index_object } )

                % create time intervals
                intervals_tof{ index_object } = math.interval( t_tof_lbs, t_tof_ubs );

            end % for index_object = 1:numel( setups )

            % avoid cell array for single setup
            if isscalar( setups )
                intervals_tof = intervals_tof{ 1 };
            end

        end % function intervals_tof = times_of_flight( setups, varargin )

        %------------------------------------------------------------------
        % spatial discretization
        %------------------------------------------------------------------
        function objects_out = discretize( setups, options_spatial )
% TODO: various types of discretization / parameter objects

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % ensure class discretizations.options_spatial_grid
            if ~isa( options_spatial, 'discretizations.options_spatial_grid')
                errorStruct.message     = 'options_spatial must be discretizations.options_spatial_grid!';
                errorStruct.identifier	= 'discretize:NoOptionsSpatialGrid';
                error( errorStruct );
            end

            %--------------------------------------------------------------
            % 2.) discretize transducer arrays and fields of view
            %--------------------------------------------------------------
            % vector or cell array of discretized apertures
% TODO: c_avg correct?
            discretizations_elements = discretize( [ setups.xdc_array ], [ setups.c_avg ], [ options_spatial.options_elements ] );

            % matrix of discretized fields of view
            discretization_FOV = reshape( discretize( [ setups.FOV ], [ options_spatial.options_FOV ] ), size( setups ) );

            %--------------------------------------------------------------
            % 3.) construct spatial discretizations
            %--------------------------------------------------------------
            % TODO: check symmetry of setups and choose class accordingly
            if 1 %issymmetric( discretizations_elements, discretization_FOV )
                N_points_per_pitch_axis = round( setups.xdc_array.element_pitch_axis ./ options_spatial.options_FOV.values( 1:(end - 1) ) );
                objects_out = discretizations.spatial_grid_symmetric( [ setups.absorption_model ], [ setups.str_name ], discretizations_elements, discretization_FOV, N_points_per_pitch_axis );
            else
                objects_out = discretizations.spatial_grid( [ setups.absorption_model ], [ setups.str_name ], discretizations_elements, discretization_FOV );
            end

        end % function objects_out = discretize( setups, options_spatial )

    end % methods

end % classdef setup
