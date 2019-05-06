%
% superclass for all reception settings w/o mixing
%
% author: Martin F. Schiffner
% date: 2019-02-05
% modified: 2019-05-04
%
classdef setting_rx_identity < controls.setting_rx

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% methods
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	methods

        %------------------------------------------------------------------
        % constructor
        %------------------------------------------------------------------
        function objects = setting_rx_identity( setup, interval_t, interval_f )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % ensure class pulse_echo_measurements.setup
            if ~isa( setup, 'pulse_echo_measurements.setup' ) || numel( setup ) ~= 1
                errorStruct.message     = 'setup must be a single pulse_echo_measurements.setup!';
                errorStruct.identifier	= 'setting_rx_identity:NoSetup';
                error( errorStruct );
            end

            % ensure equal number of dimensions and sizes
            auxiliary.mustBeEqualSize( interval_t, interval_f );

            %--------------------------------------------------------------
            % 1.) lower and upper bounds on the times-of-flight
            %--------------------------------------------------------------
%             intervals_tof = times_of_flight( setup );

            %--------------------------------------------------------------
            % 2.) create reception settings
            %--------------------------------------------------------------
            % create cell arrays
            indices_active = cell( 1, setup.xdc_array.N_elements );
            impulse_responses = cell( 1, setup.xdc_array.N_elements );

            for index_element = 1:setup.xdc_array.N_elements

                %----------------------------------------------------------
                % a) one array element is active
                %----------------------------------------------------------
                indices_active{ index_element } = index_element;

                %----------------------------------------------------------
                % b) impulse responses are identities
                %----------------------------------------------------------
                axis_t = math.sequence_increasing_regular( 0, 0, setup.T_clk );
                if setup.xdc_array.N_dimensions == 2
                    samples = physical_values.volt_per_newton_second( 1 );
                elseif setup.xdc_array.N_dimensions == 1
                    samples = physical_values.volt_meter_per_newton_second( 1 );
                else
                    errorStruct.message     = 'setup must be a single pulse_echo_measurements.setup!';
                    errorStruct.identifier	= 'setting_rx_identity:NoSetup';
                    error( errorStruct );
                end
                impulse_responses{ index_element } = discretizations.signal_array( axis_t, samples );

            end % for index_element = 1:setup.xdc_array.N_elements

            % create time intervals
            intervals_t = repmat( interval_t, [ 1, setup.xdc_array.N_elements ] );

            % create frequency intervals
            intervals_f = repmat( interval_f, [ 1, setup.xdc_array.N_elements ] );

            %--------------------------------------------------------------
            % 3.) constructor of superclass
            %--------------------------------------------------------------
            objects@controls.setting_rx( indices_active, impulse_responses, intervals_t, intervals_f );

        end % function objects = setting_rx_identity( setup, interval_t, interval_f )

	end % methods

end % classdef setting_rx_identity < pulse_echo_measurements.setting
