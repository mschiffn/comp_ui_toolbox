%
% superclass for all pulse-echo measurement settings
%
% author: Martin F. Schiffner
% date: 2019-02-05
% modified: 2019-06-01
%
classdef setting

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% properties
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	properties (SetAccess = private)

        % independent properties
        tx ( 1, 1 ) controls.setting_tx     % synthesis settings
        rx ( :, : ) controls.setting_rx     % mixer settings

        % dependent properties
        interval_hull_t ( 1, 1 ) math.interval	% hull of all recording time intervals
        interval_hull_f ( 1, 1 ) math.interval	% hull of all frequency intervals

    end % properties

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% methods
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
            % repeat default pulse-echo measurement setting
            objects = repmat( objects, size( settings_tx ) );

            % iterate pulse-echo measurement settings
            for index_object = 1:numel( settings_tx )

                % set independent properties
                objects( index_object ).tx = settings_tx( index_object );
                objects( index_object ).rx = settings_rx{ index_object };

                % set dependent properties
                [ objects( index_object ).interval_hull_t, objects( index_object ).interval_hull_f ] = hulls( objects( index_object ).rx );

            end % for index_object = 1:numel( settings_tx )

        end % function objects = setting( settings_tx, settings_rx )

        %------------------------------------------------------------------
        % spectral discretization
        %------------------------------------------------------------------
        function objects_out = discretize( settings, options_spectral )
% TODO: various types of discretization / regular vs irregular

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % ensure class discretizations.options_spectral
            if ~( isa( options_spectral, 'discretizations.options_spectral' ) && isscalar( options_spectral ) )
                errorStruct.message = 'options_spectral must be a single discretizations.options_spectral!';
                errorStruct.identifier = 'discretize:NoOptionsSpectral';
                error( errorStruct );
            end

            %--------------------------------------------------------------
            % 2.) discretize frequency intervals
            %--------------------------------------------------------------
            % specify cell arrays for settings_rx and settings_tx
            settings_rx = cell( size( settings ) );
            settings_tx = cell( size( settings ) );

            % check spectral discretization options
            switch options_spectral

                case discretizations.options_spectral.signal

                    %------------------------------------------------------
                    % a) individual frequency axis for each recorded signal
                    %------------------------------------------------------
                    % iterate pulse-echo measurement settings
                    for index_object = 1:numel( settings )
% TODO: quantize intervals_t
% TODO: check method to determine Fourier coefficients

                        % time and frequency intervals of each recorded signal
                        intervals_t = reshape( [ settings( index_object ).rx.interval_t ], size( settings( index_object ).rx ) );
                        intervals_f = reshape( [ settings( index_object ).rx.interval_f ], size( settings( index_object ).rx ) );

                        % discretize rx and tx settings
                        settings_rx{ index_object } = discretize( settings( index_object ).rx );
                        settings_tx{ index_object } = discretize( settings( index_object ).tx, intervals_t, intervals_f );

                    end % for index_object = 1:numel( settings )

                case discretizations.options_spectral.setting

                    %------------------------------------------------------
                    % b) common frequency axis for all recorded signals per setting
                    %------------------------------------------------------
                    % iterate pulse-echo measurement settings
                    for index_object = 1:numel( settings )

                        % extract unique deltas from current transducer control settings
                        deltas_unique = unique_deltas( settings( index_object ) );

                        % largest delta_unique must be integer multiple of smaller deltas_unique
                        delta_unique_max = max( deltas_unique );
                        mustBeInteger( delta_unique_max ./ deltas_unique );

                        % quantize hull of all recording time intervals using largest delta
                        interval_hull_t_quantized = quantize( settings( index_object ).interval_hull_t, delta_unique_max );

                        % discretize rx and tx settings
                        settings_rx{ index_object } = discretize( settings( index_object ).rx, interval_hull_t_quantized, settings( index_object ).interval_hull_f );
                        settings_tx{ index_object } = discretize( settings( index_object ).tx, interval_hull_t_quantized, settings( index_object ).interval_hull_f );

                    end % for index_object = 1:numel( settings )

                case discretizations.options_spectral.sequence

                    %------------------------------------------------------
                    % c) common frequency axis for all recorded signals
                    %------------------------------------------------------
                    % extract unique deltas from all transducer control settings
                    deltas_unique = unique_deltas( settings );

                    % largest delta_unique must be integer multiple of smaller deltas_unique
                    delta_unique_max = max( deltas_unique );
                    mustBeInteger( delta_unique_max ./ deltas_unique );

                    % determine hulls of all time and frequency intervals
% TODO: inject custom recording time interval
                    [ interval_hull_t, interval_hull_f ] = hulls( settings );

                    % quantize hull of all recording time intervals using delta_unique_max
                    interval_hull_t_quantized = quantize( interval_hull_t, delta_unique_max );

                    % iterate pulse-echo measurement settings
                    for index_object = 1:numel( settings )

                        % discretize rx and tx settings
                        settings_rx{ index_object } = discretize( settings( index_object ).rx, interval_hull_t_quantized, interval_hull_f );
                        settings_tx{ index_object } = discretize( settings( index_object ).tx, interval_hull_t_quantized, interval_hull_f );

                    end % for index_object = 1:numel( settings )

            end % switch options_spectral

            %--------------------------------------------------------------
            % 3.) create spectral discretizations
            %--------------------------------------------------------------
            objects_out = discretizations.spectral_points( settings_tx, settings_rx );

        end % function objects_out = discretize( settings, options_spectral )

        %------------------------------------------------------------------
        % unique deltas
        %------------------------------------------------------------------
        function deltas_unique = unique_deltas( settings )

            % extract unique deltas from all transducer control settings
            deltas_unique_tx = unique_deltas( [ settings.tx ] );
% TODO: reat out of rx might cause problems if sizes are different
            deltas_unique_rx = unique_deltas( [ settings.rx ] );
            deltas_unique = unique( [ deltas_unique_tx, deltas_unique_rx ] );

        end % function deltas_unique = unique_deltas( settings )

        %------------------------------------------------------------------
        % convex hulls of all intervals
        %------------------------------------------------------------------
        function [ interval_hull_t, interval_hull_f ] = hulls( settings )

            % convex hull of all recording time intervals
            interval_hull_t = hull( [ settings.interval_hull_t ] );

            % convex hull of all frequency intervals
            interval_hull_f = hull( [ settings.interval_hull_f ] );

        end % function [ interval_hull_t, interval_hull_f ] = hulls( settings )

	end % methods

end % classdef setting
