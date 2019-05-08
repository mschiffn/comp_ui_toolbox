%
% superclass for all spectral discretizations based on pointwise sampling
%
% author: Martin F. Schiffner
% date: 2019-02-25
% modified: 2019-05-08
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
        indices_f_to_unique
        indices_active_rx_unique ( 1, : ) double
        indices_active_rx_to_unique
        axis_k_tilde_unique ( 1, 1 ) math.sequence_increasing % axis of complex-valued wavenumbers

    end % properties

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% methods
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	methods

        %------------------------------------------------------------------
        % constructor
        %------------------------------------------------------------------
        function objects = spectral_points( settings_tx, settings_rx, absorption_model )

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

            % ensure class absorption_models.absorption_model (scalar)
            if ~( isa( absorption_model, 'absorption_models.absorption_model' ) && isscalar( absorption_model ) )
                errorStruct.message = 'absorption_model must be absorption_models.absorption_model!';
                errorStruct.identifier = 'spectral_points:NoAbsorptionModel';
                error( errorStruct );
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
                if ~( numel( settings_tx{ index_object } ) == 1 || numel( settings_tx{ index_object } ) == numel( settings_rx{ index_object } ) )
                    errorStruct.message = sprintf( 'Number of elements in settings_tx{ %d } must be one or match settings_rx{ %d }!', index_object, index_object );
                    errorStruct.identifier = 'spectral_points:SizeMismatch';
                    error( errorStruct );
                end

                % set independent properties
                objects( index_object ).tx = settings_tx{ index_object };
                objects( index_object ).rx = settings_rx{ index_object };

                % set dependent properties
                [ objects( index_object ).tx_unique, ~, objects( index_object ).indices_f_to_unique ] = unique( objects( index_object ).tx );
                [ objects( index_object ).indices_active_rx_unique, objects( index_object ).indices_active_rx_to_unique ] = unique_indices_active( objects( index_object ).rx );
                objects( index_object ).axis_k_tilde_unique = compute_wavenumbers( absorption_model, objects( index_object ).tx_unique.excitation_voltages.axis );

            end % for index_object = 1:numel( settings_tx )

        end % function objects = spectral_points( settings_tx, settings_rx, absorption_model )

        %------------------------------------------------------------------
        % get unique frequencies
        %------------------------------------------------------------------
        function axes_f_unique = get_axes_f_unique( spectral_points )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            settings_tx = reshape( [ spectral_points.tx_unique ], size( spectral_points ) );
            excitation_voltages = reshape( [ settings_tx.excitation_voltages ], size( spectral_points ) );
            axes_f_unique = reshape( [ excitation_voltages.axis ], size( spectral_points ) );

        end % function axes_f_unique = get_axes_f_unique( spectral_points )

	end % methods

end % classdef spectral_points < discretizations.spectral
