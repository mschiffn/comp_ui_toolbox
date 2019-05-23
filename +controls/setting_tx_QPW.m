%
% superclass for all synthesis settings for quasi-plane waves (QPWs)
%
% author: Martin F. Schiffner
% date: 2019-01-21
% modified: 2019-05-21
%
classdef setting_tx_QPW < controls.setting_tx

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
        function objects = setting_tx_QPW( setup, excitation_voltages_common, e_theta )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % ensure class pulse_echo_measurements.setup (scalar)
            if ~( isa( setup, 'pulse_echo_measurements.setup' ) && isscalar( setup ) )
                errorStruct.message = 'setup must be a single pulse_echo_measurements.setup!';
                errorStruct.identifier = 'setting_tx_QPW:NoSetup';
                error( errorStruct );
            end

            % excitation_voltages_common will be checked in superclass

            % ensure class math.unit_vector
            if ~isa( e_theta, 'math.unit_vector' )
                errorStruct.message = 'e_theta must be math.unit_vector!';
                errorStruct.identifier = 'setting_tx_QPW:NoUnitVector';
                error( errorStruct );
            end

            % ensure equal number of dimensions and sizes
            auxiliary.mustBeEqualSize( excitation_voltages_common, e_theta );

            %--------------------------------------------------------------
            % 2.) compute synthesis settings for QPWs
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
                % b) impulse responses are delays
                %----------------------------------------------------------
                % compute time delays for each preferred direction of propagation
                time_delays_act = e_theta( index_object ).components * centers( setup.xdc_array )' / setup.c_avg;
                time_delays_act = time_delays_act - min( time_delays_act );

                % specify impulse responses
                indices_q = round( time_delays_act / setup.T_clk );
                impulse_responses{ index_object } = discretizations.delta_matrix( indices_q, setup.T_clk, physical_values.meter_per_volt_second( ones( size( indices_q ) ) ) );

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

                % preferred direction of propagation
                objects( index_object ).e_theta = e_theta( index_object );

            end

        end % function objects = setting_tx_QPW( setup, excitation_voltages_common, e_theta )

	end % methods

end % classdef setting_tx_QPW < controls.setting_tx
