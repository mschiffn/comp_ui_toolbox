%
% superclass for all sequential pulse-echo measurements
%
% author: Martin F. Schiffner
% date: 2019-01-14
% modified: 2019-04-22
%
classdef sequence

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% properties
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	properties ( SetAccess = private )

        % independent properties
        setup %( 1, 1 ) pulse_echo_measurements.setup        % pulse-echo measurement setup
        settings %( :, : ) pulse_echo_measurements.setting	% pulse-echo measurement settings

        % dependent properties
        interval_t ( 1, 1 ) math.interval	% hull of all recording time intervals
        interval_f ( 1, 1 ) math.interval	% hull of all frequency intervals

    end % properties

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% methods
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	methods ( Access = public )

        %------------------------------------------------------------------
        % constructor
        %------------------------------------------------------------------
        function objects = sequence( setups, settings )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % ensure class pulse_echo_measurements.setup
            if ~isa( setups, 'pulse_echo_measurements.setup' )
                errorStruct.message     = 'setups must be pulse_echo_measurements.setup!';
                errorStruct.identifier	= 'sequence:NoSetup';
                error( errorStruct );
            end

            % ensure cell array for settings
            if ~iscell( settings )
                settings = { settings };
            end

            % multiple setups / single settings
            if ~isscalar( setups ) && isscalar( settings )
                settings = repmat( settings, size( setups ) );
            end

            % single setups / multiple settings
            if isscalar( setups ) && ~isscalar( settings )
                setups = repmat( setups, size( settings ) );
            end

            % ensure equal number of dimensions and sizes
            auxiliary.mustBeEqualSize( setups, settings );

            %--------------------------------------------------------------
            % 2.) create sequences of pulse-echo measurements
            %--------------------------------------------------------------
            % repeat default sequence
            objects = repmat( objects, size( setups ) );

            % iterate sequences
            for index_object = 1:numel( objects )

                % ensure class pulse_echo_measurements.setting
                if ~isa( settings{ index_object }, 'pulse_echo_measurements.setting' )
                    errorStruct.message     = 'settings must be pulse_echo_measurements.setting!';
                    errorStruct.identifier	= 'sequence:NoSetting';
                    error( errorStruct );
                end

                % TODO: ensure that settings are compatible w/ setup

                % set independent properties
                objects( index_object ).setup = setups( index_object );
                objects( index_object ).settings = settings{ index_object };

                % set dependent properties
% TODO: quantize time interval
                [ objects( index_object ).interval_t, objects( index_object ).interval_f ] = hulls( objects( index_object ).settings );

            end % for index_object = 1:numel( objects )

        end % function objects = sequence( setups, settings )

        
        %------------------------------------------------------------------
        % spatiospectral discretizations
        %------------------------------------------------------------------
        function spatiospectrals = discretize( sequences, options )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % ensure class discretizations.options
            if ~isa( options, 'discretizations.options' )
                errorStruct.message     = 'options must be discretizations.options!';
                errorStruct.identifier	= 'discretize:NoOptions';
                error( errorStruct );
            end

            % ensure equal number of dimensions and sizes
            auxiliary.mustBeEqualSize( sequences, options );

            %--------------------------------------------------------------
            % 2.) spatial discretizations
            %--------------------------------------------------------------
            setups = reshape( [ sequences.setup ], size( sequences ) );
            spatials = discretize( setups, [ options.spatial ] );

            %--------------------------------------------------------------
            % 3.) spectral discretizations
            %--------------------------------------------------------------
            % specify cell array for spectrals
            spectrals = cell( size( sequences ) );

            % iterate sequential pulse-echo measurements
            for index_object = 1:numel( sequences )

                % discretize pulse-echo measurement settings
                spectrals{ index_object } = discretize( sequences( index_object ).settings, options( index_object ).spectral );

            end % for index_object = 1:numel( sequences )

            %--------------------------------------------------------------
            % 4.) create spatiospectral discretizations
            %--------------------------------------------------------------
            spatiospectrals = discretizations.spatiospectral( spatials, spectrals );

        end % function spatiospectrals = discretize( sequences, options )

	end % methods

end % classdef sequence
