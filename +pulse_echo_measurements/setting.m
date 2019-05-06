%
% superclass for all pulse-echo measurement settings
%
% author: Martin F. Schiffner
% date: 2019-02-05
% modified: 2019-04-22
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

            % iterate pulse-echo measurement settings
            for index_object = 1:numel( settings_tx )

                % set independent properties
                objects( index_object ).tx = settings_tx( index_object );
                objects( index_object ).rx = settings_rx{ index_object };

                % set dependent properties
                [ objects( index_object ).interval_t, objects( index_object ).interval_f ] = hulls( objects( index_object ).rx );

            end % for index_object = 1:numel( settings_tx )

        end % function objects = setting( settings_tx, settings_rx )

        %------------------------------------------------------------------
        % spectral discretization
        %------------------------------------------------------------------
        function objects_out = discretize( settings, absorption_models, options_spectral )
% TODO: various types of discretization / regular vs irregular

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % ensure cell array for settings
            if ~iscell( settings )
                settings = { settings };
            end

            % class absorption_models.absorption_model is ensured by discretizations.spectral_points
            % class discretizations.options_spectral is ensured by switch statement

            % multiple settings / single absorption_models
            if ~isscalar( settings ) && isscalar( absorption_models )
                absorption_models = repmat( absorption_models, size( settings ) );
            end

            % multiple settings / single options_spectral
            if ~isscalar( settings ) && isscalar( options_spectral )
                options_spectral = repmat( options_spectral, size( settings ) );
            end

            % ensure equal number of dimensions and sizes
            auxiliary.mustBeEqualSize( settings, absorption_models, options_spectral );

            %--------------------------------------------------------------
            % 2.) discretize frequency intervals
            %--------------------------------------------------------------
            % iterate pulse-echo measurement settings
            for index_object = 1:numel( settings )

                % initialize cell arrays
                settings_rx = cell( size( settings{ index_object } ) );
                settings_tx = cell( size( settings{ index_object } ) );

                switch options_spectral( index_object )

                    case discretizations.options_spectral.signal

                        %--------------------------------------------------
                        % a) individual frequency axis for each recorded signal
                        %--------------------------------------------------
                        for index_setting = 1:numel( settings{ index_object } )

                            % time and frequency intervals of each recorded signal
                            intervals_t = reshape( [ settings{ index_object }( index_setting ).rx.interval_t ], size( settings{ index_object }( index_setting ).rx ) );
                            intervals_f = reshape( [ settings{ index_object }( index_setting ).rx.interval_f ], size( settings{ index_object }( index_setting ).rx ) );

                            % discretize rx and tx settings
                            settings_rx{ index_setting } = discretize( settings{ index_object }( index_setting ).rx );
                            settings_tx{ index_setting } = discretize( settings{ index_object }( index_setting ).tx, intervals_t, intervals_f );

                        end % for index_setting = 1:numel( settings{ index_object } )

                    case discretizations.options_spectral.setting

                        %--------------------------------------------------
                        % b) common frequency axis for all recorded signals per setting
                        %--------------------------------------------------
                        for index_setting = 1:numel( settings{ index_object } )

                            % discretize rx and tx settings
                            settings_rx{ index_setting } = discretize( settings{ index_object }( index_setting ).rx, settings{ index_object }( index_setting ).interval_t, settings{ index_object }( index_setting ).interval_f );
                            settings_tx{ index_setting } = discretize( settings{ index_object }( index_setting ).tx, settings{ index_object }( index_setting ).interval_t, settings{ index_object }( index_setting ).interval_f );

                        end % for index_setting = 1:numel( settings{ index_object } )

                    case discretizations.options_spectral.sequence

                        %--------------------------------------------------
                        % c) common frequency axis for all recorded signals
                        %--------------------------------------------------
                        % determine hulls of all time and frequency intervals
                        [ interval_t_hull, interval_f_hull ] = hulls( settings{ index_object } );

                        % iterate settings
                        for index_setting = 1:numel( settings{ index_object } )

                            % discretize rx and tx settings
                            settings_rx{ index_setting } = discretize( settings{ index_object }( index_setting ).rx, interval_t_hull, interval_f_hull );
                            settings_tx{ index_setting } = discretize( settings{ index_object }( index_setting ).tx, interval_t_hull, interval_f_hull );

                        end % for index_setting = 1:numel( settings{ index_object } )

                    otherwise

                        %--------------------------------------------------
                        % d) unknown spectral discretization method
                        %--------------------------------------------------
                        errorStruct.message = sprintf( 'options_spectral( %d ) must be discretizations.options_spectral!', index_object );
                        errorStruct.identifier	= 'discretize:NoOptionsSpectral';
                        error( errorStruct );

                end % switch options_spectral( index_object )

                %----------------------------------------------------------
                % 3.) create spectral discretizations
                %----------------------------------------------------------
% TODO: vectorize function
                objects_out{ index_object } = discretizations.spectral_points( settings_tx, settings_rx, absorption_models( index_object ) );

            end % for index_object = 1:numel( settings )

            % avoid cell array for single spectral discretization
            if isscalar( objects_out )
                objects_out = objects_out{ 1 };
            end

        end % function objects_out = discretize( settings, absorption_models, options_spectral )

        %------------------------------------------------------------------
        % convex hulls of all intervals
        %------------------------------------------------------------------
        function [ interval_hull_t, interval_hull_f ] = hulls( settings )

            % convex hull of all recording time intervals
            interval_hull_t = hull( [ settings.interval_t ] );

            % convex hull of all frequency intervals
            interval_hull_f = hull( [ settings.interval_f ] );

        end % function [ interval_hull_t, interval_hull_f ] = hulls( settings )

        %------------------------------------------------------------------
        % compute hash values
        %------------------------------------------------------------------
        function str_hash = hash( settings )

            % specify cell array for str_hash
            str_hash = cell( size( settings ) );

            % iterate pulse-echo measurement settings
            for index_object = 1:numel( settings )

                % use DataHash function to compute hash value
                str_hash{ index_object } = auxiliary.DataHash( settings( index_object ) );

            end % for index_object = 1:numel( settings )

            % avoid cell array for single pulse-echo measurement setting
            if isscalar( settings )
                str_hash = str_hash{ 1 };
            end

        end % function str_hash = hash( settings )

	end % methods

end % classdef setting
