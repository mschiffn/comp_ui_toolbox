%
% superclass for all synthesis settings for superpositions of randomly-apodized quasi-(d-1)-spherical waves
%
% author: Martin F. Schiffner
% date: 2019-01-26
% modified: 2019-03-28
%
classdef setting_rnd_apo_del < syntheses.setting

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % properties
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	properties (SetAccess = private)

        % independent properties
        setting_rng_apo ( 1, 1 ) auxiliary.setting_rng      % settings of the random number generator
        setting_rng_del ( 1, 1 ) auxiliary.setting_rng      % settings of the random number generator
        e_theta ( 1, 1 ) math.unit_vector        % preferred direction of propagation (1)
    end % properties

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % methods
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    methods

        %------------------------------------------------------------------
        % constructor
        %------------------------------------------------------------------
        function objects = setting_rnd_apo_del( setup, excitation_voltages_common, e_theta, settings_rng_apo, settings_rng_del )

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

            % excitation_voltages_common will be checked in superclass

            % ensure class math.unit_vector
            if ~isa( e_theta, 'math.unit_vector' )
                errorStruct.message     = 'e_theta must be math.unit_vector!';
                errorStruct.identifier	= 'setting_rnd_del:NoUnitVectors';
                error( errorStruct );
            end

            % ensure class auxiliary.setting_rng
            if ~isa( settings_rng_apo, 'auxiliary.setting_rng' ) || ~isa( settings_rng_del, 'auxiliary.setting_rng' )
                errorStruct.message     = 'settings_rng_apo and settings_rng_del must be auxiliary.setting_rng!';
                errorStruct.identifier	= 'setting_rnd_apo:NoSettingRng';
                error( errorStruct );
            end
            % assertion: settings_rng_apo and settings_rng_del are auxiliary.setting_rng

            % ensure equal number of dimensions and sizes
            auxiliary.mustBeEqualSize( excitation_voltages_common, e_theta, settings_rng_apo, settings_rng_del );

            %--------------------------------------------------------------
            % 2.) compute synthesis settings for superpositions of randomly-apodized and randomly-delayed quasi-(d-1)-spherical waves
            %--------------------------------------------------------------
            % number of sequential syntheses
            N_objects = numel( excitation_voltages_common );

            % allocate cell arrays to store synthesis settings
            indices_active = cell( size( excitation_voltages_common ) );
            apodization_weights = cell( size( excitation_voltages_common ) );
            time_delays = cell( size( excitation_voltages_common ) );
            excitation_voltages = cell( size( excitation_voltages_common ) );

            for index_object = 1:N_objects

                %----------------------------------------------------------
                % a) all array elements are active
                %----------------------------------------------------------
                indices_active{ index_object } = (1:setup.xdc_array.N_elements);

                %----------------------------------------------------------
                % b) random apodization weights
                %----------------------------------------------------------
                % seed random number generator
                rng( settings_rng_apo( index_object ).seed, settings_rng_apo( index_object ).str_name );

                % sample Bernoulli distribution
                apodization_weights_act = rand( 1, setup.xdc_array.N_elements );
                indicator = (apodization_weights_act >= 0.5);
                apodization_weights_act(1, indicator) = 1;
                apodization_weights_act(1, ~indicator) = -1;

                apodization_weights{ index_object } = apodization_weights_act;

                %----------------------------------------------------------
                % c) random time delays
                %----------------------------------------------------------
                % seed random number generator
                rng( settings_rng_del( index_object ).seed, settings_rng_del( index_object ).str_name );

                % compute permissible maximum time shift
                t_shift_max = sum( ( setup.xdc_array.N_elements_axis( 1:(setup.FOV.N_dimensions - 1) ) - 1 ) .* setup.xdc_array.element_pitch_axis( 1:(setup.FOV.N_dimensions - 1) ) .* abs( e_theta( index_object ).components( 1:(setup.FOV.N_dimensions - 1) ) ), 2 ) / setup.c_avg;
                T_inc = t_shift_max / ( setup.xdc_array.N_elements - 1 );

                % compute random time delays
                time_delays_act = ( randperm( setup.xdc_array.N_elements ) - 1 ) * T_inc;
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
                objects( index_object ).setting_rng_apo = settings_rng_apo( index_object );
                objects( index_object ).setting_rng_del = settings_rng_del( index_object );
                objects( index_object ).e_theta = e_theta( index_object );
            end

        end % function objects = setting_rnd_apo_del( setup, excitation_voltages_common, e_theta, settings_rng_apo, settings_rng_del )

	end % methods

end % classdef setting_rnd_apo_del
