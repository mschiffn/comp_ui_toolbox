%
% superclass for all reception settings w/o mixing
%
% author: Martin F. Schiffner
% date: 2019-02-05
% modified: 2019-10-22
%
classdef setting_rx_identity < scattering.sequences.settings.controls.setting_rx

	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	%% methods
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	methods

        %------------------------------------------------------------------
        % constructor
        %------------------------------------------------------------------
        function objects = setting_rx_identity( setup, setting_tx, interval_f )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % ensure class scattering.sequences.setups.setup (scalar)
            if ~( isa( setup, 'scattering.sequences.setups.setup' ) && isscalar( setup ) )
                errorStruct.message = 'setup must be a single scattering.sequences.setups.setup!';
                errorStruct.identifier = 'setting_rx_identity:NoSingleSetup';
                error( errorStruct );
            end

            % ensure class scattering.sequences.settings.controls.setting_tx (scalar)
            if ~( isa( setting_tx, 'scattering.sequences.settings.controls.setting_tx' ) && isscalar( setting_tx ) )
                errorStruct.message = 'setting_tx must be a single scattering.sequences.settings.controls.setting_tx!';
                errorStruct.identifier = 'setting_rx_identity:NoSingleSettingTx';
                error( errorStruct );
            end

            %--------------------------------------------------------------
            % 2.) create reception settings
            %--------------------------------------------------------------
            % specify active array elements
            indices_active = num2cell( ( 1:setup.xdc_array.N_elements ) );

            % impulse responses are identities
            weight = physical_values.volt_per_newton * physical_values.meter^( 3 - setup.FOV.shape.N_dimensions );
            impulse_responses = repmat( { discretizations.delta_matrix( 0, setup.T_clk, weight ) }, [ 1, setup.xdc_array.N_elements ] );

            %--------------------------------------------------------------
            % 3.) compute recording time intervals
            %--------------------------------------------------------------
            % initialize lower and upper bounds on the support
            t_lbs = physical_values.second( zeros( 1, setup.xdc_array.N_elements ) );
            t_ubs = physical_values.second( zeros( 1, setup.xdc_array.N_elements ) );

            % iterate mixed voltage signals
            for index_mix = 1:setup.xdc_array.N_elements

                % allocate memory
                t_lbs_all = physical_values.time( zeros( numel( setting_tx.indices_active ), 1 ) );
                t_ubs_all = physical_values.time( zeros( numel( setting_tx.indices_active ), 1 ) );

                % iterate active tx elements
                for index_active_tx = 1:numel( setting_tx.indices_active )

                    index_element_tx = setting_tx.indices_active( index_active_tx );

                    % support of excitation_voltages
                    if isa( setting_tx.excitation_voltages, 'discretizations.signal' )
                        t_lb_tx_act = setting_tx.excitation_voltages( index_active_tx ).axis.members( 1 );
                        t_ub_tx_act = setting_tx.excitation_voltages( index_active_tx ).axis.members( end );
                    else
                        indicator = double( abs( setting_tx.excitation_voltages.samples( :, index_active_tx ) ) ) >= eps;
                        members = setting_tx.excitation_voltages.axis.members( indicator );
                        t_lb_tx_act = members( 1 );
                        t_ub_tx_act = members( end );
                    end

                    % support of impulse responses
                    if isa( setting_tx.impulse_responses, 'discretizations.signal' )
                        t_lb_tx_act = t_lb_tx_act + setting_tx.impulse_responses( index_active_tx ).axis.members( 1 );
                        t_ub_tx_act = t_ub_tx_act + setting_tx.impulse_responses( index_active_tx ).axis.members( end );
                    else
                        indicator = double( abs( setting_tx.impulse_responses.samples( :, index_active_tx ) ) ) >= eps;
                        members = setting_tx.impulse_responses.axis.members( indicator );
                        t_lb_tx_act = t_lb_tx_act + members( 1 );
                        t_ub_tx_act = t_ub_tx_act + members( end );
                    end

                    % compute lower and upper bounds on the recording time intervals
                    t_lbs_all( index_active_tx ) = t_lb_tx_act + setup.intervals_tof( index_element_tx, index_mix ).lb;
                    t_ubs_all( index_active_tx ) = t_ub_tx_act + setup.intervals_tof( index_element_tx, index_mix ).ub;

                end % for index_active_tx = 1:numel( setting_tx.indices_active )

                t_lbs( index_mix ) = min( t_lbs_all );
                t_ubs( index_mix ) = max( t_ubs_all );

            end % for index_mix = 1:setup.xdc_array.N_elements

            % create time intervals for all mixes
            intervals_t = math.interval( t_lbs, t_ubs );

            % create frequency intervals
            intervals_f = repmat( interval_f, [ 1, setup.xdc_array.N_elements ] );

            %--------------------------------------------------------------
            % 4.) constructor of superclass
            %--------------------------------------------------------------
            objects@scattering.sequences.settings.controls.setting_rx( indices_active, impulse_responses, intervals_t, intervals_f );

        end % function objects = setting_rx_identity( setup, setting_tx, interval_f )

	end % methods

end % classdef setting_rx_identity < scattering.sequences.settings.controls.setting_rx
