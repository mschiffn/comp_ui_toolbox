%
% superclass for all synthesis settings for superpositions of randomly-delayed quasi-(d-1)-spherical waves
%
% author: Martin F. Schiffner
% date: 2019-01-26
% modified: 2019-08-25
%
classdef setting_tx_rnd_del < controls.setting_tx

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% properties
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	properties (SetAccess = private)

        % independent properties
        setting_rng ( 1, 1 ) auxiliary.setting_rng      % settings of the random number generator
        e_theta ( 1, 1 ) math.unit_vector               % preferred direction of propagation for permutation of delays

    end % properties

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% methods
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	methods

        %------------------------------------------------------------------
        % constructor
        %------------------------------------------------------------------
        function objects = setting_tx_rnd_del( setup, u_tx_tilde, e_theta, settings_rng )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % ensure class pulse_echo_measurements.setup (scalar)
            if ~( isa( setup, 'pulse_echo_measurements.setup' ) && isscalar( setup ) )
                errorStruct.message = 'setup must be a single pulse_echo_measurements.setup!';
                errorStruct.identifier = 'setting_tx_rnd_del:NoSetup';
                error( errorStruct );
            end

            % ensure class transducers.array_planar_regular_orthogonal
            if ~isa( setup.xdc_array, 'transducers.array_planar_regular_orthogonal' )
                errorStruct.message = 'setup.xdc_array must be transducers.array_planar_regular_orthogonal!';
                errorStruct.identifier = 'setting_tx_rnd_del:NoRegularOrthogonalTransducerArray';
                error( errorStruct );
            end

            % superclass checks u_tx_tilde

            % ensure class math.unit_vector
            if ~isa( e_theta, 'math.unit_vector' )
                errorStruct.message = 'e_theta must be math.unit_vector!';
                errorStruct.identifier = 'setting_tx_rnd_del:NoUnitVector';
                error( errorStruct );
            end

            % ensure class auxiliary.setting_rng
            if ~isa( settings_rng, 'auxiliary.setting_rng' )
                errorStruct.message = 'settings_rng must be auxiliary.setting_rng!';
                errorStruct.identifier = 'setting_tx_rnd_del:NoSettingRng';
                error( errorStruct );
            end

            % ensure equal number of dimensions and sizes
            auxiliary.mustBeEqualSize( u_tx_tilde, e_theta, settings_rng );

            %--------------------------------------------------------------
            % 2.) compute synthesis settings for superpositions of randomly-delayed quasi-(d-1)-spherical waves
            %--------------------------------------------------------------
            % number of sequential syntheses
            N_objects = numel( u_tx_tilde );

            % specify cell arrays to store synthesis settings
            indices_active = cell( size( u_tx_tilde ) );
            impulse_responses = cell( size( u_tx_tilde ) );
            excitation_voltages = cell( size( u_tx_tilde ) );

            % iterate synthesis settings
            for index_object = 1:N_objects

                %----------------------------------------------------------
                % a) all array elements are active
                %----------------------------------------------------------
                indices_active{ index_object } = (1:setup.xdc_array.N_elements);

                %----------------------------------------------------------
                % b) impulse responses are random time delays
                %----------------------------------------------------------
                % seed random number generator
                rng( settings_rng( index_object ).seed, settings_rng( index_object ).str_name );

                % compute permissible maximum time shift
                N_dimensions_lateral = setup.FOV.shape.N_dimensions - 1;
                t_shift_max = sum( ( setup.xdc_array.N_elements_axis( 1:N_dimensions_lateral )' - 1 ) .* setup.xdc_array.cell_ref.edge_lengths( 1:N_dimensions_lateral ) .* abs( e_theta( index_object ).components( 1:N_dimensions_lateral ) ), 2 ) / setup.homogeneous_fluid.c_avg;
                % incorrect value for reproduction of old results: T_inc = t_shift_max / setup.xdc_array.N_elements;
                T_inc = t_shift_max / ( setup.xdc_array.N_elements - 1 );
%                 T_inc = physical_values.second( 2e-6 );	% test using full pulse length

                % compute random time delays
                time_delays_act = ( randperm( setup.xdc_array.N_elements ) - 1 ) * T_inc;

                % specify impulse responses
                indices_q = round( time_delays_act / setup.T_clk );
                impulse_responses{ index_object } = discretizations.delta_matrix( indices_q, setup.T_clk, physical_values.meter_per_volt_second( ones( size( indices_q ) ) ) );

                %----------------------------------------------------------
                % c) identical excitation voltages for all array elements
                %----------------------------------------------------------
                excitation_voltages{ index_object } = u_tx_tilde( index_object );

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
                objects( index_object ).e_theta = e_theta( index_object );
            end

        end % function objects = setting_tx_rnd_del( setup, u_tx_tilde, e_theta, settings_rng )

	end % methods

end % classdef setting_tx_rnd_del < controls.setting_tx
