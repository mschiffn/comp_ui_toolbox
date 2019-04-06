%
% superclass for all pulse-echo measurement settings
%
% author: Martin F. Schiffner
% date: 2019-02-05
% modified: 2019-04-04
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
        interval_t ( 1, 1 ) math.interval	% hull of all recording time intervals
        interval_f ( 1, 1 ) math.interval	% hull of all frequency intervals

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

            % set and check independent properties
            for index_object = 1:numel( settings_tx )

                % set independent properties
                objects( index_object ).tx = settings_tx( index_object );
                objects( index_object ).rx = settings_rx{ index_object };

                % set dependent properties
                objects( index_object ).interval_t = interval_t_hull( objects( index_object ) );
                objects( index_object ).interval_f = interval_f_hull( objects( index_object ) );

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
            % TODO: multiple settings / single options_spectral
            if ~isscalar( settings ) && isscalar( options_spectral )
            end

            %--------------------------------------------------------------
            % 2.) discretize frequency intervals
            %--------------------------------------------------------------
            % initialize cell arrays
            spectral_points_rx = cell( size( settings ) );
            spectral_points_tx = cell( size( settings ) );

            switch options_spectral

                case discretizations.options_spectral.signal

                    %------------------------------------------------------
                    % a) individual discretization for each recorded signal
                    %------------------------------------------------------
                    for index_setting = 1:numel( settings )

                        % time and frequency intervals of each recorded signal
                        intervals_t = reshape( [ settings( index_setting ).rx.interval_t ], size( settings( index_setting ).rx ) );
                        intervals_f = reshape( [ settings( index_setting ).rx.interval_f ], size( settings( index_setting ).rx ) );

                        % discretize rx and tx settings
                        spectral_points_rx{ index_setting } = discretize( settings( index_setting ).rx );
                        spectral_points_tx{ index_setting } = discretize( settings( index_setting ).tx, intervals_t, intervals_f );

                    end % for index_setting = 1:numel( settings )

                case discretizations.options_spectral.setting

                    %------------------------------------------------------
                    % b) common discretization for all recorded signals per setting
                    %------------------------------------------------------
                    for index_setting = 1:numel( settings )

                        % discretize rx and tx settings
                        spectral_points_rx{ index_setting } = discretize( settings( index_setting ).rx, settings( index_setting ).interval_t, settings( index_setting ).interval_f );
                        spectral_points_tx{ index_setting } = discretize( settings( index_setting ).tx, settings( index_setting ).interval_t, settings( index_setting ).interval_f );

                    end % for index_setting = 1:numel( settings )

                case discretizations.options_spectral.sequence

                    %------------------------------------------------------
                    % c) common discretization for all recorded signals
                    %------------------------------------------------------
                    % determine hull time and frequency intervals
                    interval_t_hull = hull( [ settings.interval_t ] );
                    interval_f_hull = hull( [ settings.interval_f ] );

                    % iterate settings
                    for index_setting = 1:numel( settings )

                        % discretize rx and tx settings
                        spectral_points_rx{ index_setting } = discretize( settings( index_setting ).rx, interval_t_hull, interval_f_hull );
                        spectral_points_tx{ index_setting } = discretize( settings( index_setting ).tx, interval_t_hull, interval_f_hull );

                    end % for index_setting = 1:numel( settings )

                otherwise

                    %------------------------------------------------------
                    % d) unknown spectral discretization method
                    %------------------------------------------------------
                    errorStruct.message     = 'options_spectral must be discretizations.options_spectral!';
                    errorStruct.identifier	= 'discretize:NoOptionsSpectral';
                    error( errorStruct );

            end % switch options_spectral

            %--------------------------------------------------------------
            % 3.) create spectral discretizations
            %--------------------------------------------------------------
            objects_out = discretizations.spectral_points( spectral_points_tx, spectral_points_rx );

        end % function objects_out = discretize( settings, options_spectral )

        %------------------------------------------------------------------
        % convex hull of all recording time intervals
        %------------------------------------------------------------------
        function objects_out = interval_t_hull( objects_in )

            % create objects_out of equal size
            objects_out = repmat( objects_in( 1 ).rx( 1 ).interval_t, size( objects_in ) );

            % iterate objects_in
            for index_object = 1:numel( objects_in )

                objects_out( index_object ) = hull( [ objects_in( index_object ).rx.interval_t ] );

            end % for index_object = 1:numel( objects_in )

        end % function objects_out = interval_t_hull( objects_in )

        %------------------------------------------------------------------
        % convex hull of all frequency intervals
        %------------------------------------------------------------------
        function objects_out = interval_f_hull( objects_in )

            % create objects_out of equal size
            objects_out = repmat( objects_in( 1 ).rx( 1 ).interval_f, size( objects_in ) );

            % iterate objects_in
            for index_object = 1:numel( objects_in )

                objects_out( index_object ) = hull( [ objects_in( index_object ).rx.interval_f ] );

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
