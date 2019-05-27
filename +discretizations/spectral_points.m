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

                % ensure correct number of settings_tx{ index_object }
                if ~( isscalar( settings_tx{ index_object } ) || numel( settings_tx{ index_object } ) == numel( settings_rx{ index_object } ) )
                    errorStruct.message = sprintf( 'Number of elements in settings_tx{ %d } must be one or match settings_rx{ %d }!', index_object, index_object );
                    errorStruct.identifier = 'spectral_points:SizeMismatch';
                    error( errorStruct );
                end

                % ensure identical frequency axes
% TODO: stimmt so nicht! -> vergleich per setting notwendig
                temp = [ settings_tx{ index_object }.impulse_responses, settings_rx{ index_object }.impulse_responses ];
                if ~isequal( temp.axis )
                    errorStruct.message = sprintf( 'Excitation voltages and impulse responses in settings_tx( %d ) and settings_rx( %d ) must have identical axes!', index_object, index_object );
                    errorStruct.identifier = 'spectral_points:AxesMismatch';
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
                % determine tx settings for unique frequencies
                [ objects( index_object ).tx_unique, ~, objects( index_object ).indices_f_to_unique ] = unique( objects( index_object ).tx );
                if isscalar( objects( index_object ).tx )
                    objects( index_object ).indices_f_to_unique = repmat( objects( index_object ).indices_f_to_unique, size( objects( index_object ).rx ) );
                end
                [ objects( index_object ).indices_active_rx_unique, objects( index_object ).indices_active_rx_to_unique ] = unique_indices_active( objects( index_object ).rx );

                % numbers of observations in each mixed voltage signal
                objects( index_object ).N_observations = compute_N_observations( objects( index_object ).rx );

                % compute normal velocities (unique frequencies)
                objects( index_object ).v_d_unique = compute_normal_velocities( objects( index_object ).tx_unique );

            end % for index_object = 1:numel( settings_tx )

        end % function objects = spectral_points( settings_tx, settings_rx )

	end % methods

end % classdef spectral_points < discretizations.spectral
