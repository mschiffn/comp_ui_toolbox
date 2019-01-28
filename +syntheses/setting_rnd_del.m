%
% superclass for all synthesis settings for superpositions of randomly-delayed quasi-(d-1)-spherical waves
%
% author: Martin F. Schiffner
% date: 2019-01-26
% modified: 2019-01-28
%
classdef setting_rnd_del < syntheses.setting

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % properties
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	properties (SetAccess = private)

        % independent properties
        setting_rng ( 1, 1 ) syntheses.setting_rng	% settings of the random number generator
        e_theta ( 1, : ) double     % preferred directions of propagation (1)
    end % properties

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % methods
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    methods

        %------------------------------------------------------------------
        % constructor
        %------------------------------------------------------------------
        function objects = setting_rnd_del( setup, excitation_voltages_common, e_theta, settings_rng )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % ensure class pulse_echo_measurements.setup
            if ~isa( setup, 'pulse_echo_measurements.setup' ) || numel( setup ) ~= 1
                errorStruct.message     = 'setup must be a single pulse_echo_measurements.setup!';
                errorStruct.identifier	= 'setting_rnd_del:NoSetup';
                error( errorStruct );
            end
            % assertion: setup is a single pulse_echo_measurements.setup

            % ensure class syntheses.setting_rng
            if ~isa( settings_rng, 'syntheses.setting_rng' )
                errorStruct.message     = 'settings_rng must be syntheses.setting_rng!';
                errorStruct.identifier	= 'setting_rnd_apo:NoSettingRng';
                error( errorStruct );
            end
            % assertion: settings_rng is syntheses.setting_rng

            % ensure equal number of elements
            if numel( excitation_voltages_common ) ~= numel( settings_rng )
                errorStruct.message     = 'Number of elements in excitation_voltages_common has to match that in settings_rng!';
                errorStruct.identifier	= 'setting_rnd_apo:DimensionMismatch';
                error( errorStruct );
            end

            % excitation_voltages_common will be checked in superclass

            %--------------------------------------------------------------
            % 2.) compute synthesis settings for superpositions of randomly-delayed quasi-(d-1)-spherical waves
            %--------------------------------------------------------------
            % number of sequential syntheses
            N_objects = size( excitation_voltages_common, 1 );

            % allocate cell arrays to store synthesis settings
            indices_active = cell( N_objects, 1 );
            apodization_weights = cell( N_objects, 1 );
            time_delays = cell( N_objects, 1 );
            excitation_voltages = cell( N_objects, 1 );

            % iterate objects
            for index_object = 1:N_objects

                %----------------------------------------------------------
                % a) all array elements are active
                %----------------------------------------------------------
                indices_active{ index_object } = (1:setup.xdc_array.N_elements);

                %----------------------------------------------------------
                % b) unity apodization weights
                %----------------------------------------------------------
                apodization_weights{ index_object } = physical_values.apodization_weight( ones( 1, setup.xdc_array.N_elements ) );

                %----------------------------------------------------------
                % c) random time delays
                %----------------------------------------------------------
                % seed random number generator
                rng( settings_rng( index_object ).seed, settings_rng( index_object ).str_name );

                % compute permissible maximum time shift
                t_shift_max = sum( abs( e_theta( index_object, 1:(setup.FOV.N_dimensions - 1) ) ) .* setup.xdc_array.element_pitch_axis( 1:(setup.FOV.N_dimensions - 1) ) .* ( setup.xdc_array.N_elements_axis( 1:(setup.FOV.N_dimensions - 1) ) - 1 ), 2 ) / setup.c_avg;

                % compute random time delays
                time_delays_act = t_shift_max * ( randperm( setup.xdc_array.N_elements ) - 1 ) / ( setup.xdc_array.N_elements - 1 );
                time_delays{ index_object } = physical_values.time( time_delays_act );

                %----------------------------------------------------------
                % d) identical excitation voltages for all array elements
                %----------------------------------------------------------
                excitation_voltages{ index_object } = repmat( excitation_voltages_common( index_object ), [ 1, setup.xdc_array.N_elements ] );

            end % for index_object = 1:N_objects

            %--------------------------------------------------------------
            % 3.) constructor of superclass
            %--------------------------------------------------------------
            objects@syntheses.setting( indices_active, apodization_weights, time_delays, excitation_voltages );

            %--------------------------------------------------------------
            % 4.) set independent properties
            %--------------------------------------------------------------
            for index_object = 1:N_objects
                objects( index_object ).setting_rng = settings_rng( index_object );
                objects( index_object ).e_theta = e_theta( index_object, : );
            end

        end % function objects = setting_rnd_del( setup, excitation_voltages_common, e_theta, settings_rng )

	end % methods

end % classdef setting_rnd_del
