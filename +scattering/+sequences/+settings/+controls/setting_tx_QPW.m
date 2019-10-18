%
% superclass for all quasi-plane wave (QPWs) synthesis settings
%
% author: Martin F. Schiffner
% date: 2019-01-21
% modified: 2019-10-17
%
classdef setting_tx_QPW < scattering.sequences.settings.controls.setting_tx

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% properties
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	properties (SetAccess = private)

        % independent properties
        e_theta ( 1, 1 ) math.unit_vector	% preferred direction of propagation

    end % properties

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% methods
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	methods

        %------------------------------------------------------------------
        % constructor
        %------------------------------------------------------------------
        function objects = setting_tx_QPW( setup, u_tx_tilde, e_theta )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % ensure class scattering.sequences.setups.setup (scalar)
            if ~( isa( setup, 'scattering.sequences.setups.setup' ) && isscalar( setup ) )
                errorStruct.message = 'setup must be a single scattering.sequences.setups.setup!';
                errorStruct.identifier = 'setting_tx_QPW:NoSetup';
                error( errorStruct );
            end

            % ensure class scattering.sequences.setups.transducers.array_planar_regular_orthogonal
            if ~isa( setup.xdc_array, 'scattering.sequences.setups.transducers.array_planar_regular_orthogonal' )
                errorStruct.message = 'setup.xdc_array must be scattering.sequences.setups.transducers.array_planar_regular_orthogonal!';
                errorStruct.identifier = 'setting_tx_QPW:NoRegularOrthogonalTransducerArray';
                error( errorStruct );
            end

            % superclass checks u_tx_tilde

            % ensure class math.unit_vector
            if ~isa( e_theta, 'math.unit_vector' )
                errorStruct.message = 'e_theta must be math.unit_vector!';
                errorStruct.identifier = 'setting_tx_QPW:NoUnitVector';
                error( errorStruct );
            end

            % ensure equal number of dimensions and sizes
            auxiliary.mustBeEqualSize( u_tx_tilde, e_theta );

            %--------------------------------------------------------------
            % 2.) compute synthesis settings for QPWs
            %--------------------------------------------------------------
            % number of sequential syntheses
            N_objects = numel( u_tx_tilde );

            % allocate cell arrays to store synthesis settings
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
                % b) impulse responses are delays
                %----------------------------------------------------------
                % compute time delays for each preferred direction of propagation
                time_delays_act = e_theta( index_object ).components * [ setup.xdc_array.positions_ctr, zeros( setup.xdc_array.N_elements, 1 ) ]' / setup.homogeneous_fluid.c_avg;
                time_delays_act = time_delays_act - min( time_delays_act );

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
            objects@scattering.sequences.settings.controls.setting_tx( indices_active, impulse_responses, excitation_voltages );

            %--------------------------------------------------------------
            % 4.) set independent properties
            %--------------------------------------------------------------
            for index_object = 1:N_objects

                % preferred direction of propagation
                objects( index_object ).e_theta = e_theta( index_object );

            end

        end % function objects = setting_tx_QPW( setup, u_tx_tilde, e_theta )

	end % methods

end % classdef setting_tx_QPW < scattering.sequences.settings.controls.setting_tx
