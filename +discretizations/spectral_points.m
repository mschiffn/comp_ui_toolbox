%
% superclass for all spectral discretizations based on pointwise sampling
%
% author: Martin F. Schiffner
% date: 2019-02-25
% modified: 2019-05-13
%
classdef spectral_points < discretizations.spectral

	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	%% properties
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	properties (SetAccess = private)

        % independent properties
        tx ( :, : ) controls.setting_tx
        rx ( :, : ) controls.setting_rx

        % dependent properties
        tx_unique ( :, : ) controls.setting_tx
        v_d_unique ( :, 1 ) discretizations.signal_matrix       % normal velocities (unique frequencies)
        indices_f_to_unique
        indices_active_rx_unique ( 1, : ) double
        indices_active_rx_to_unique
        N_observations ( :, : ) double                          % numbers of observations in each mixed voltage signal

    end % properties

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% methods
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	methods

        %------------------------------------------------------------------
        % constructor
        %------------------------------------------------------------------
        function objects = spectral_points( settings_tx, settings_rx )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % ensure cell array for settings_tx
            if ~iscell( settings_tx )
                settings_tx = { settings_tx };
            end

            % ensure cell array for settings_rx
            if ~iscell( settings_rx )
                settings_rx = { settings_rx };
            end

            % ensure equal number of dimensions and sizes
            auxiliary.mustBeEqualSize( settings_tx, settings_rx );

            %--------------------------------------------------------------
            % 2.) constructor of superclass
            %--------------------------------------------------------------
            objects@discretizations.spectral();
            objects = repmat( objects, size( settings_tx ) );

            %--------------------------------------------------------------
            % 3.) set independent and dependent properties
            %--------------------------------------------------------------
            for index_object = 1:numel( settings_tx )

% TODO: ensure identical frequency axes ?

                % ensure correct number of settings_tx{ index_object }
                if ~( isscalar( settings_tx{ index_object } ) || numel( settings_tx{ index_object } ) == numel( settings_rx{ index_object } ) )
                    errorStruct.message = sprintf( 'Number of elements in settings_tx{ %d } must be one or match settings_rx{ %d }!', index_object, index_object );
                    errorStruct.identifier = 'spectral_points:SizeMismatch';
                    error( errorStruct );
                end

                %----------------------------------------------------------
                % a) set independent properties
                %----------------------------------------------------------
                objects( index_object ).tx = settings_tx{ index_object };
                objects( index_object ).rx = settings_rx{ index_object };

                %----------------------------------------------------------
                % b) set dependent properties
                %----------------------------------------------------------
                [ objects( index_object ).tx_unique, ~, objects( index_object ).indices_f_to_unique ] = unique( objects( index_object ).tx );
                [ objects( index_object ).indices_active_rx_unique, objects( index_object ).indices_active_rx_to_unique ] = unique_indices_active( objects( index_object ).rx );

                % numbers of observations in each mixed voltage signal
                objects( index_object ).N_observations = compute_N_observations( objects( index_object ) );

                % compute normal velocities (unique frequencies)
                objects( index_object ).v_d_unique = compute_normal_velocities( objects( index_object ) );

            end % for index_object = 1:numel( settings_tx )

        end % function objects = spectral_points( settings_tx, settings_rx )

        %------------------------------------------------------------------
        % compute normal velocities (unique frequencies)
        %------------------------------------------------------------------
        function v_d_unique = compute_normal_velocities( spectral_points )

            % extract transducer control settings for each sequential pulse-echo measurement (unique frequencies)
            settings_tx_unique = reshape( [ spectral_points.tx_unique ], size( spectral_points ) );

            % extract excitation voltages and transfer functions for each sequential pulse-echo measurement (unique frequencies)
            excitation_voltages = reshape( [ settings_tx_unique.excitation_voltages ], size( spectral_points ) );
            impulse_responses = reshape( [ settings_tx_unique.impulse_responses ], size( spectral_points ) );

            % compute velocities for each sequential pulse-echo measurement (unique frequencies)
            v_d_unique = excitation_voltages .* impulse_responses;

        end % function v_d_unique = compute_normal_velocities( spectral_points )

        %------------------------------------------------------------------
        % compute numbers of observations
        %------------------------------------------------------------------
        function N_observations = compute_N_observations( spectral_points )

            % specify cell array for N_observations
            N_observations = cell( size( spectral_points ) );

            % iterate spectral discretizations
            for index_object = 1:numel( spectral_points )

                N_observations{ index_object } = cellfun( @numel, spectral_points( index_object ).indices_f_to_unique );

            end % for index_object = 1:numel( spectral_points )

            % avoid cell array for single spectral_points
            if isscalar( spectral_points )
                N_observations = N_observations{ 1 };
            end

        end % function N_observations = compute_N_observations( spectral_points )

	end % methods

end % classdef spectral_points < discretizations.spectral
