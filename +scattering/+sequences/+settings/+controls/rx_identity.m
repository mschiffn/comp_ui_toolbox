%
% superclass for all reception settings w/o mixing
%
% author: Martin F. Schiffner
% date: 2019-02-05
% modified: 2019-11-28
%
classdef rx_identity < scattering.sequences.settings.controls.rx

	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	%% methods
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	methods

        %------------------------------------------------------------------
        % constructor
        %------------------------------------------------------------------
        function objects = rx_identity( setup, setting_tx, intervals_f, indices_active, impulse_responses )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % ensure class scattering.sequences.setups.setup (scalar)
            if ~( isa( setup, 'scattering.sequences.setups.setup' ) && isscalar( setup ) )
                errorStruct.message = 'setup must be a single scattering.sequences.setups.setup!';
                errorStruct.identifier = 'rx_identity:NoSingleSetup';
                error( errorStruct );
            end

            % ensure class scattering.sequences.settings.controls.tx (scalar)
            if ~( isa( setting_tx, 'scattering.sequences.settings.controls.tx' ) && isscalar( setting_tx ) )
                errorStruct.message = 'setting_tx must be a single scattering.sequences.settings.controls.tx!';
                errorStruct.identifier = 'rx_identity:NoSingleSettingTx';
                error( errorStruct );
            end

            % ensure nonempty indices_active
            if nargin < 4 || isempty( indices_active )
                % specify active array elements
                indices_active = num2cell( ( 1:setup.xdc_array.N_elements ) );
            end

            % ensure cell array for indices_active
            if ~iscell( indices_active )
                indices_active = num2cell( indices_active );
            end

            % superclass ensures finite positive integers for indices_active

            % ensure nonempty impulse_responses
            if nargin < 5 || isempty( impulse_responses )
                % impulse responses are identities
                weight = physical_values.volt_per_newton * physical_values.meter^( 3 - setup.FOV.shape.N_dimensions );
                impulse_responses = repmat( { discretizations.delta_matrix( 0, setup.T_clk, weight ) }, [ 1, setup.xdc_array.N_elements ] );
            end

            % ensure cell array for impulse_responses
            if ~iscell( impulse_responses )
                impulse_responses = num2cell( impulse_responses );
            end

            % superclass ensures class discretizations.signal_matrix for impulse_responses

            % multiple indices_active / single intervals_f
            if ~isscalar( indices_active ) && isscalar( intervals_f )
                intervals_f = repmat( intervals_f, size( indices_active ) );
            end

            % ensure equal number of dimensions and sizes
            auxiliary.mustBeEqualSize( indices_active, intervals_f, impulse_responses );

            %--------------------------------------------------------------
            % 2.) compute recording time intervals
            %--------------------------------------------------------------
            % initialize lower and upper bounds on the support
            t_lbs = physical_values.second( zeros( 1, numel( indices_active ) ) );
            t_ubs = physical_values.second( zeros( 1, numel( indices_active ) ) );

            % iterate mixed voltage signals
            for index_active_rx = 1:numel( indices_active )

                index_element_rx = indices_active{ index_active_rx };

                % support of impulse responses
                t_lb_rx_act = impulse_responses{ index_active_rx }.axis.members( 1 );
                t_ub_rx_act = impulse_responses{ index_active_rx }.axis.members( end );

                % allocate memory
                t_lbs_all = physical_values.second( zeros( numel( setting_tx.indices_active ), 1 ) );
                t_ubs_all = physical_values.second( zeros( numel( setting_tx.indices_active ), 1 ) );

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
                    t_lbs_all( index_active_tx ) = t_lb_rx_act + t_lb_tx_act + setup.intervals_tof( index_element_tx, index_element_rx ).lb;
                    t_ubs_all( index_active_tx ) = t_ub_rx_act + t_ub_tx_act + setup.intervals_tof( index_element_tx, index_element_rx ).ub;

                end % for index_active_tx = 1:numel( setting_tx.indices_active )

                t_lbs( index_active_rx ) = min( t_lbs_all );
                t_ubs( index_active_rx ) = max( t_ubs_all );

            end % for index_active_rx = 1:numel( indices_active )

            % create time intervals for all mixes
            intervals_t = math.interval( t_lbs, t_ubs );

            %--------------------------------------------------------------
            % 3.) create reception settings w/o mixing
            %--------------------------------------------------------------
            % constructor of superclass
            objects@scattering.sequences.settings.controls.rx( indices_active, impulse_responses, intervals_t, intervals_f );

        end % function objects = rx_identity( setup, setting_tx, intervals_f, indices_active, impulse_responses )

	end % methods

end % classdef rx_identity < scattering.sequences.settings.controls.rx
