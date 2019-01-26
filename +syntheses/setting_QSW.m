%
% superclass for all synthesis settings for quasi-(d-1)-spherical waves
%
% author: Martin F. Schiffner
% date: 2019-01-21
% modified: 2019-01-25
%
classdef setting_QSW < syntheses.setting

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % properties
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	properties (SetAccess = private)

        % independent properties
        positions_src ( 1, : ) double	% positions of the virtual sources (m)
        angles ( 1, : ) double          % aperture angles (rad) 0 <= angles < pi/2
        c_avg ( 1, 1 ) double           % average small-signal sound speed (m/s)
    end % properties

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % methods
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    methods

        %------------------------------------------------------------------
        % constructor
        %------------------------------------------------------------------
        function objects = setting_QSW( setup, excitation_voltages_common, positions_src, angles )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % ensure class pulse_echo_measurements.setup
            if ~isa( setup, 'pulse_echo_measurements.setup' ) || numel( setup ) ~= 1
                errorStruct.message     = 'setup must be a single pulse_echo_measurements.setup!';
                errorStruct.identifier	= 'setting_QPW:NoSetup';
                error( errorStruct );
            end
            % assertion: setup is a single pulse_echo_measurements.setup

            % excitation_voltages_common will be checked in superclass

            %--------------------------------------------------------------
            % 2.) compute synthesis settings for QSWs
            %--------------------------------------------------------------
            % number of sequential syntheses
            N_objects = size( positions_src, 1 );

            % allocate cell arrays to store synthesis settings
            indices_active = cell( N_objects, 1 );
            apodization_weights = cell( N_objects, 1 );
            time_delays = cell( N_objects, 1 );
            excitation_voltages = cell( N_objects, 1 );

            for index_object = 1:N_objects

                %----------------------------------------------------------
                % a) determine active array elements
                %----------------------------------------------------------
                vectors_src_ctr = setup.xdc_array.grid_ctr.positions - repmat( positions_src( index_object, : ), [ setup.xdc_array.N_elements, 1 ] );
                distances_src_ctr = sqrt( sum( abs( vectors_src_ctr ).^2, 2 ) );
                indicator_distance = distances_src_ctr >= eps;
                indicator_active = false( 1, setup.xdc_array.N_elements );
                indicator_active( ~indicator_distance ) = true;
                indicator_active( indicator_distance ) = asin( abs( vectors_src_ctr( indicator_distance, 1 ) ./ distances_src_ctr( indicator_distance ) ) ) <= angles( index_object, 1 ) / 2;
                indices_active{ index_object } = find( indicator_active == true );
                N_elements_active = numel( indices_active{ index_object } );

                %----------------------------------------------------------
                % b) unity apodization weights
                %----------------------------------------------------------
                apodization_weights{ index_object } = physical_values.apodization_weight( ones( 1, N_elements_active ) );

                %----------------------------------------------------------
                % c) compute time delays for each virtual source
                %----------------------------------------------------------
                time_delays_act = distances_src_ctr( indicator_active ) / setup.c_avg;
                time_delays_act = time_delays_act - min( time_delays_act );
                time_delays{ index_object } = physical_values.time( time_delays_act );

                %----------------------------------------------------------
                % d) identical excitation voltages for all array elements
                %----------------------------------------------------------
                excitation_voltages{ index_object } = repmat( excitation_voltages_common( index_object ), [ 1, N_elements_active ] );

            end % for index_object = 1:N_objects

            %--------------------------------------------------------------
            % 3.) constructor of superclass
            %--------------------------------------------------------------
            objects@syntheses.setting( indices_active, apodization_weights, time_delays, excitation_voltages );

            %--------------------------------------------------------------
            % 4.) set independent properties
            %--------------------------------------------------------------
            for index_object = 1:N_objects
                objects( index_object ).positions_src = positions_src( index_object, : );
                objects( index_object ).angles = angles( index_object, : );
                objects( index_object ).c_avg = setup.c_avg;
            end

        end % function objects = setting_QSW( setup, excitation_voltages_common, positions_src, angles )

	end % methods

end % classdef setting_QSW
