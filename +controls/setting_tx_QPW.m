%
% superclass for all synthesis settings for quasi-plane waves (QPWs)
%
% author: Martin F. Schiffner
% date: 2019-01-21
% modified: 2019-02-06
%
classdef setting_QPW < syntheses.setting

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % properties
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	properties (SetAccess = private)

        % independent properties
        e_theta ( 1, 1 ) physical_values.unit_vector	% preferred direction of propagation (1)
    end % properties

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % methods
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	methods

        %------------------------------------------------------------------
        % constructor
        %------------------------------------------------------------------
        function objects = setting_QPW( setup, excitation_voltages_common, e_theta )

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

            % ensure class physical_values.unit_vector
            if ~isa( e_theta, 'physical_values.unit_vector' )
                errorStruct.message     = 'e_theta must be physical_values.unit_vector!';
                errorStruct.identifier	= 'setting_rnd_del:NoUnitVectors';
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
                % b) unity apodization weights
                %----------------------------------------------------------
                apodization_weights{ index_object } = physical_values.apodization_weight( ones( 1, setup.xdc_array.N_elements ) );

                %----------------------------------------------------------
                % c) compute time delays for each preferred direction of propagation
                %----------------------------------------------------------
                time_delays_act = e_theta( index_object ).components * double( centers( setup.xdc_array ) )' / setup.c_avg;
                time_delays_act = time_delays_act - min( time_delays_act );
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
                objects( index_object ).e_theta = e_theta( index_object );
            end

        end % function objects = setting_QPW( setup, excitation_voltages_common, e_theta )

	end % methods

end % classdef setting_QPW
