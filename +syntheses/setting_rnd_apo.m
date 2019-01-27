%
% superclass for all synthesis settings for superpositions of randomly-apodized quasi-(d-1)-spherical waves
%
% author: Martin F. Schiffner
% date: 2019-01-26
% modified: 2019-01-26
%
classdef setting_rnd_apo < syntheses.setting

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % properties
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	properties (SetAccess = private)

        % independent properties
        rng_seed ( 1, 1 ) double	% seed for random number generator
        rng_name ( 1, : ) char      % name of random number generator
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

            %--------------------------------------------------------------
            % 2.) compute synthesis settings for superpositions of randomly-apodized quasi-(d-1)-spherical waves
            %--------------------------------------------------------------
            % number of sequential syntheses
            N_objects = size( excitation_voltages_common, 1 );

            % allocate cell arrays to store synthesis settings
            indices_active = cell( N_objects, 1 );
            apodization_weights = cell( N_objects, 1 );
            time_delays = cell( N_objects, 1 );
            excitation_voltages = cell( N_objects, 1 );

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

                apodization_weights{ index_object } = physical_values.apodization_weight( apodization_weights_act );

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
                objects( index_object ).rng_seed = rng_seeds( index_object );
                objects( index_object ).rng_name = rng_name( index_object );
            end

        end % function objects = setting_rnd_apo( setup, excitation_voltages_common, rng_seeds, rng_name )

	end % methods

end % classdef setting_rnd_apo
