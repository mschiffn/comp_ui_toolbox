%
% superclass for all synthesis settings for superpositions of randomly-apodized quasi-(d-1)-spherical waves
%
% author: Martin F. Schiffner
% date: 2019-01-26
% modified: 2019-05-10
%
classdef setting_tx_rnd_apo < controls.setting_tx

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% properties
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	properties (SetAccess = private)

        % independent properties
        setting_rng ( 1, 1 ) auxiliary.setting_rng      % settings of the random number generator

    end % properties

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% methods
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	methods

        %------------------------------------------------------------------
        % constructor
        %------------------------------------------------------------------
        function objects = setting_tx_rnd_apo( setup, excitation_voltages_common, settings_rng )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % ensure class pulse_echo_measurements.setup (scalar)
            if ~( isa( setup, 'pulse_echo_measurements.setup' ) && isscalar( setup ) )
                errorStruct.message     = 'setup must be a single pulse_echo_measurements.setup!';
                errorStruct.identifier	= 'setting_tx_QPW:NoSetup';
                error( errorStruct );
            end

            % excitation_voltages_common will be checked in superclass

            % ensure class auxiliary.setting_rng
            if ~isa( settings_rng, 'auxiliary.setting_rng' )
                errorStruct.message     = 'settings_rng must be auxiliary.setting_rng!';
                errorStruct.identifier	= 'setting_tx_rnd_apo:NoSettingRng';
                error( errorStruct );
            end

            % ensure equal number of dimensions and sizes
            auxiliary.mustBeEqualSize( excitation_voltages_common, settings_rng );

            %--------------------------------------------------------------
            % 2.) compute synthesis settings for superpositions of randomly-apodized quasi-(d-1)-spherical waves
            %--------------------------------------------------------------
            % number of sequential syntheses
            N_objects = numel( excitation_voltages_common );

            % allocate cell arrays to store synthesis settings
            indices_active = cell( size( excitation_voltages_common ) );
            impulse_responses = cell( size( excitation_voltages_common ) );
            excitation_voltages = cell( size( excitation_voltages_common ) );

            % iterate synthesis settings
            for index_object = 1:N_objects

                %----------------------------------------------------------
                % a) all array elements are active
                %----------------------------------------------------------
                indices_active{ index_object } = (1:setup.xdc_array.N_elements);

                %----------------------------------------------------------
                % b) impulse responses are random apodization weights
                %----------------------------------------------------------
                % seed random number generator
                rng( settings_rng( index_object ).seed, settings_rng( index_object ).str_name );

                % sample Bernoulli distribution
                apodization_weights_act = rand( setup.xdc_array.N_elements, 1 );
                indicator = ( apodization_weights_act >= 0.5 );
                apodization_weights_act( indicator ) = 1;
                apodization_weights_act( ~indicator ) = -1;

                % specify impulse responses
                axis_t = math.sequence_increasing_regular( 0, 0, setup.T_clk );
                samples = physical_values.meter_per_volt_squaresecond( apodization_weights_act );

                impulse_responses{ index_object } = discretizations.signal_matrix( axis_t, samples );

                %----------------------------------------------------------
                % c) identical excitation voltages for all array elements
                %----------------------------------------------------------
                excitation_voltages{ index_object } = excitation_voltages_common( index_object );

            end % for index_object = 1:N_objects

            %--------------------------------------------------------------
            % 3.) constructor of superclass
            %--------------------------------------------------------------
            objects@controls.setting_tx( indices_active, impulse_responses, excitation_voltages );

            %--------------------------------------------------------------
            % 4.) set independent properties
            %--------------------------------------------------------------
            for index_object = 1:N_objects
                objects( index_object ).setting_rng = settings_rng( index_object );
            end

        end % function objects = setting_tx_rnd_apo( setup, excitation_voltages_common, settings_rng )

	end % methods

end % classdef setting_tx_rnd_apo < controls.setting_tx
