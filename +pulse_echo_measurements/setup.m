%
% superclass for all pulse-echo measurement setups
%
% author: Martin F. Schiffner
% date: 2018-03-12
% modified: 2019-03-28
%
classdef setup < handle

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % properties
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	properties (SetAccess = private)

        % independent properties
        xdc_array ( 1, 1 ) transducers.array = transducers.L14_5_38     % transducer array
        FOV ( 1, 1 ) fields_of_view.field_of_view                       % field of view
        % TODO: properties of the lossy homogeneous fluid
        absorption_model ( 1, 1 ) absorption_models.absorption_model = absorption_models.time_causal( 0, 0.5, 1, 1540, 4e6, 1 )	% absorption model for the lossy homogeneous fluid
        c_avg = physical_values.meter_per_second( 1500 );               % average small-signal sound speed
        T_clk = physical_values.second( 1 / 80e6 );                     % time period of the clock signal
        str_name = 'default'                                            % name

    end % properties

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % methods
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	methods

        %------------------------------------------------------------------
        % constructor
        %------------------------------------------------------------------
        function object = setup( xdc_array, FOV, absorption_model, str_name )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % ensure matching number of dimensions
            if xdc_array.N_dimensions ~= ( FOV.N_dimensions - 1 )
                errorStruct.message     = 'The number of dimensions in FOV must exceed that in xdc_array by unity!';
                errorStruct.identifier	= 'array:DimensionMismatch';
                error( errorStruct );
            end

            %--------------------------------------------------------------
            % 2.) set independent properties
            %--------------------------------------------------------------
            % set independent properties
            object.xdc_array = xdc_array;
            object.FOV = FOV;
            object.absorption_model = absorption_model;
            object.str_name = str_name;

        end % function object = setup( xdc_array, FOV, absorption_model, str_name )

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
            discretizations_elements = discretize( [ setups.xdc_array ], [ options_spatial.options_elements ] );

            % matrix of discretized fields of view
            discretization_FOV = discretize( [ setups.FOV ], [ options_spatial.options_FOV ] );

            %--------------------------------------------------------------
            % 3.) construct spatial discretizations
            %--------------------------------------------------------------
            % TODO: check symmetry of setups and choose class accordingly
            if 1 %issymmetric( discretizations_elements, discretization_FOV )
                objects_out = discretizations.spatial_grid_symmetric( discretizations_elements, discretization_FOV );
            else
                objects_out = discretizations.spatial_grid( discretizations_elements, discretization_FOV );
            end

        end % function objects_out = discretize( setups, options_spatial )

        %------------------------------------------------------------------
        % lower and upper bounds on the times-of-flight
        %------------------------------------------------------------------
        function results = times_of_flight( object )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % ensure planar transducer array and FOV with orthotope shape
            if ~( isa( object.xdc_array, 'transducers.planar_transducer_array' ) && isa( object.FOV, 'fields_of_view.orthotope' ) )
                errorStruct.message     = 'Current implementation requires planar transducer array and FOV with orthotope shape!';
                errorStruct.identifier	= 'times_of_flight:NoPlanarOrOrthotope';
                error( errorStruct );
            end

            %--------------------------------------------------------------
            % 2.) estimate lower and upper bounds
            %--------------------------------------------------------------
            % vertices of the FOV
            pos_vertices = vertices( object.FOV );

            % allocate memory
            t_tof_lbs = physical_values.time( zeros( object.xdc_array.N_elements ) );
            t_tof_ubs = physical_values.time( zeros( object.xdc_array.N_elements ) );

            % investigate all pairs of array elements
            for index_element_tx = 1:object.xdc_array.N_elements

                % center coordinates of tx element
                pos_tx_ctr = object.xdc_array.grid_ctr.positions( index_element_tx );

                for index_element_rx = index_element_tx:object.xdc_array.N_elements

                    % center coordinates of rx element
                    pos_rx_ctr = object.xdc_array.grid_ctr.positions( index_element_rx );

                    % center coordinates of prolate spheroid
                    pos_spheroid_ctr = ( pos_tx_ctr + pos_rx_ctr ) / 2;

                    % distance from center coordinates to focal points
                    dist_focus_ctr = norm( pos_rx_ctr - pos_tx_ctr ) / 2;

                    %------------------------------------------------------
                    % a) lower bound on the time-of-flight
                    %------------------------------------------------------
                    t_tof_lbs( index_element_tx, index_element_rx ) = physical_values.time( 2 * sqrt( dist_focus_ctr^2 + double( object.FOV.intervals( end ).bounds( 1 ) )^2 ) / object.c_avg );
                    t_tof_lbs( index_element_rx, index_element_tx ) = t_tof_lbs( index_element_tx, index_element_rx );

                    %------------------------------------------------------
                    % b) upper bound on the time-of-flight
                    %------------------------------------------------------
                    % determine vertex of maximum distance
                    [ dist_ctr_vertices_max, index_max ] = max( sqrt( sum( ( [ pos_spheroid_ctr, 0 ] - double( pos_vertices ) ).^2, 2 ) ) );

                    % compute upper bound
                    t_tof_ubs( index_element_tx, index_element_rx ) = ( norm( double( pos_vertices( index_max, : ) ) - [ pos_tx_ctr, 0 ] ) + norm( [ pos_rx_ctr, 0 ] - double( pos_vertices( index_max, : ) ) ) ) / object.c_avg;
                    t_tof_ubs( index_element_rx, index_element_tx ) = t_tof_ubs( index_element_tx, index_element_rx );

                end % for index_element_rx = index_element_tx:object.xdc_array.N_elements
            end % for index_element_tx = 1:object.xdc_array.N_elements

            % create time intervals
            results = math.interval_time( t_tof_lbs, t_tof_ubs);

        end % function results = times_of_flight( object )

        %------------------------------------------------------------------
        % compute hash value
        %------------------------------------------------------------------
        function str_hash = hash( setups )

            % use DataHash function to compute hash value
            str_hash = auxiliary.DataHash( setups );

        end % function str_hash = hash( setups )

    end % methods

end % classdef setup
