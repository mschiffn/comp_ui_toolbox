%
% superclass for all synthesis settings for superpositions of randomly-apodized quasi-(d-1)-spherical waves
%
% author: Martin F. Schiffner
% date: 2019-01-26
% modified: 2019-02-02
%
classdef setting_rnd_apo < syntheses.setting

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % properties
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	properties (SetAccess = private)

        % independent properties
        setting_rng ( 1, 1 ) auxiliary.setting_rng      % settings of the random number generator
    end % properties

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % methods
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	methods

        %------------------------------------------------------------------
        % constructor
        %------------------------------------------------------------------
        function objects = setting_rnd_apo( setup, excitation_voltages_common, settings_rng )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % ensure class pulse_echo_measurements.setup
            if ~isa( setup, 'pulse_echo_measurements.setup' ) || numel( setup ) ~= 1
                errorStruct.message     = 'setup must be a single pulse_echo_measurements.setup!';
                errorStruct.identifier	= 'setting_rnd_apo:NoSetup';
                error( errorStruct );
            end
            % assertion: setup is a single pulse_echo_measurements.setup

            % excitation_voltages_common will be checked in superclass

            % ensure class auxiliary.setting_rng
            if ~isa( settings_rng, 'auxiliary.setting_rng' )
                errorStruct.message     = 'settings_rng must be auxiliary.setting_rng!';
                errorStruct.identifier	= 'setting_rnd_apo:NoSettingRng';
                error( errorStruct );
            end
            % assertion: settings_rng is auxiliary.setting_rng

            % ensure equal number of dimensions and sizes
            auxiliary.mustBeEqualSize( excitation_voltages_common, settings_rng );

            %--------------------------------------------------------------
            % 2.) compute synthesis settings for superpositions of randomly-apodized quasi-(d-1)-spherical waves
            %--------------------------------------------------------------
            % number of sequential syntheses
            N_objects = numel( excitation_voltages_common );

            % allocate cell arrays to store synthesis settings
            indices_active = cell( size( excitation_voltages_common ) );
            apodization_weights = cell( size( excitation_voltages_common ) );
            time_delays = cell( size( excitation_voltages_common ) );
            excitation_voltages = cell( size( excitation_voltages_common ) );

            % iterate objects
            for index_object = 1:N_objects

                %----------------------------------------------------------
                % a) all array elements are active
                %----------------------------------------------------------
                indices_active{ index_object } = (1:setup.xdc_array.N_elements);

                %----------------------------------------------------------
                % b) random apodization weights
                %----------------------------------------------------------
                % seed random number generator
                rng( settings_rng( index_object ).seed, settings_rng( index_object ).str_name );

                % sample Bernoulli distribution
                apodization_weights_act = rand( 1, setup.xdc_array.N_elements );
                indicator = (apodization_weights_act >= 0.5);
                apodization_weights_act(1, indicator) = 1;
                apodization_weights_act(1, ~indicator) = -1;

                apodization_weights{ index_object } = apodization_weights_act;

                %----------------------------------------------------------
                % c) no time delays
                %----------------------------------------------------------
                time_delays{ index_object } = physical_values.time( zeros( 1, setup.xdc_array.N_elements ) );

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
            end

        end % function objects = setting_rnd_apo( setup, excitation_voltages_common, settings_rng )

	end % methods

end % classdef setting_rnd_apo
