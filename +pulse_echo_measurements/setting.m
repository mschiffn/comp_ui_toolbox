%
% superclass for all pulse-echo measurement settings
%
% author: Martin F. Schiffner
% date: 2019-02-05
% modified: 2019-02-25
%
classdef setting

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % properties
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	properties (SetAccess = private)

        % independent properties
        tx ( 1, 1 ) controls.setting_tx     % synthesis settings
        rx ( :, : ) controls.setting_rx     % mixer settings

        % dependent properties
        interval_t ( 1, 1 ) physical_values.interval_time       % hull of all recording time intervals
        interval_f ( 1, 1 ) physical_values.interval_frequency	% hull of all frequency intervals

    end % properties

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % methods
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	methods

        %------------------------------------------------------------------
        % constructor
        %------------------------------------------------------------------
        function objects = setting( settings_tx, settings_rx )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % ensure cell array for settings_rx
            if ~iscell( settings_rx )
                settings_rx = { settings_rx };
            end

            % ensure equal number of dimensions and sizes
            auxiliary.mustBeEqualSize( settings_tx, settings_rx );

            %--------------------------------------------------------------
            % 2.) create pulse-echo measurement settings
            %--------------------------------------------------------------
            % construct objects
            objects = repmat( objects, size( settings_tx ) );

            % set and check independent properties
            for index_object = 1:numel( settings_tx )

                % set independent properties
                objects( index_object ).tx = settings_tx( index_object );
                objects( index_object ).mixes = settings_rx{ index_object };

                % set dependent properties
                objects( index_object ).interval_t = interval_t_hull( objects( index_object ) );
                objects( index_object ).interval_f = interval_f_hull( objects( index_object ) );

            end

        end % function objects = setting( settings_tx, settings_rx )

        %------------------------------------------------------------------
        % spectral discretization
        %------------------------------------------------------------------
        function objects_out = discretize( settings, options_spectral )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % ensure class discretizations.options_spectral
            if ~isa( options_spectral, 'discretizations.options_spectral')
                errorStruct.message     = 'options_spectral must be discretizations.options_spectral!';
                errorStruct.identifier	= 'discretize:NoOptionsSpectral';
                error( errorStruct );
            end

            %--------------------------------------------------------------
            % 2.) discretize frequency intervals
            %--------------------------------------------------------------
            switch options_spectral

                case discretizations.options_spectral.signal

                    %------------------------------------------------------
                    % a) individual discretization for each mix
                    %------------------------------------------------------
                    % initialize cell arrays
                    sets_f = cell( size( settings ) );
                    transfer_functions_rx = cell( size( settings ) );
                    transfer_functions_tx = cell( size( settings ) );
                    excitation_voltages = cell( size( settings ) );

                    % iterate settings
                    for index_setting = 1:numel( settings )

                        % recording time intervals and frequency intervals for each received signal
                        intervals_t = reshape( [ settings( index_setting ).rx.interval_t ], size( settings( index_setting ).rx ) );
                        intervals_f = reshape( [ settings( index_setting ).rx.interval_f ], size( settings( index_setting ).rx ) );

                        %
                        transfer_functions_rx{ index_setting } = discretize( settings( index_setting ).rx );
                        [ excitation_voltages{ index_setting }, transfer_functions_tx{ index_setting } ] = discretize( settings( index_setting ).tx, intervals_t, intervals_f );

                    end % for index_setting = 1:numel( settings )

                    % create spectral discretizations
                    objects_out = discretizations.spectral_points( sets_f, transfer_functions_rx, excitation_voltages, transfer_functions_tx );

                case discretizations.options_spectral.setting
                    % TODO: various types of discretization / regular vs irregular

                    %------------------------------------------------------
                    % b) common discretization for all mixes
                    %------------------------------------------------------

                    % compute Fourier coefficients of the impulse responses
                    for index_object = 1:numel( settings )

                        excitation_voltages = fourier_coefficients( settings( index_object ).tx.excitation_voltages, settings( index_object ).interval_t, settings( index_object ).interval_f );

                        % transfer functions for each mix
                        transfer_functions = cell( size( settings( index_object ).mixes ) );
                        for index_mix = 1:numel( settings( index_object ).mixes )

                            transfer_functions{ index_mix } = fourier_transform( settings( index_object ).mixes( index_mix ).xdc_control.impulse_responses, settings( index_object ).interval_t, settings( index_object ).interval_f );
                        end

                    end % for index_object = 1:numel( settings )

                case discretizations.options_spectral.sequence

                    %------------------------------------------------------
                    % b) common spectral discretization for all settings in the sequence
                    %------------------------------------------------------
                    % determine hull time and frequency intervals
                    interval_t_hull = hull( [ settings.interval_t ] );
                    interval_f_hull = hull( [ settings.interval_f ] );

                    % discretize hull frequency interval
                    objects_out = discretize( interval_f_hull, 1 ./ abs( interval_t_hull ) );

                otherwise

                    %------------------------------------------------------
                    % c) unknown spectral discretization method
                    %------------------------------------------------------
                    errorStruct.message     = 'discretizations.options_spectral not implemented!';
                    errorStruct.identifier	= 'discretize:UnknownMethod';
                    error( errorStruct );

            end % switch options_spectral

        end % function objects_out = discretize( settings, options_spectral )

        %------------------------------------------------------------------
        % convex hull of all recording time intervals
        %------------------------------------------------------------------
        function objects_out = interval_t_hull( objects_in )

            % create objects_out of equal size
            objects_out = repmat( objects_in( 1 ).mixes( 1 ).interval_t, size( objects_in ) );

            % iterate objects_in
            for index_object = 1:numel( objects_in )

                objects_out( index_object ) = hull( [ objects_in( index_object ).mixes.interval_t ] );

            end % for index_object = 1:numel( objects_in )

        end % function objects_out = interval_t_hull( objects_in )

        %------------------------------------------------------------------
        % convex hull of all frequency intervals
        %------------------------------------------------------------------
        function objects_out = interval_f_hull( objects_in )

            % create objects_out of equal size
            objects_out = repmat( objects_in( 1 ).mixes( 1 ).interval_f, size( objects_in ) );

            % iterate objects_in
            for index_object = 1:numel( objects_in )

                objects_out( index_object ) = hull( [ objects_in( index_object ).mixes.interval_f ] );

            end % for index_object = 1:numel( objects_in )

        end % function objects_out = interval_f_hull( objects_in )

        %------------------------------------------------------------------
        % compute hash value
        %------------------------------------------------------------------
        function str_hash = hash( object )

            % use DataHash function to compute hash value
            str_hash = auxiliary.DataHash( object );

        end % function str_hash = hash( object )

	end % methods

end % classdef setting
