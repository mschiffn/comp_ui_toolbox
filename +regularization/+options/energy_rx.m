%
% superclass for all energy options
%
% author: Martin F. Schiffner
% date: 2020-02-21
% modified: 2020-08-07
%
classdef energy_rx

	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	%% properties
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	properties (SetAccess = private)

        % independent properties
        momentary ( 1, 1 ) scattering.options.momentary { mustBeNonempty } = scattering.options.momentary                           % momentary scattering options
        tgc ( 1, 1 ) regularization.tgc.tgc { mustBeNonempty } = regularization.tgc.off                                             % TGC
        dictionary ( 1, 1 ) regularization.dictionaries.dictionary { mustBeNonempty } = regularization.dictionaries.identity        % dictionary

	end % properties

	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	%% methods
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	methods

        %------------------------------------------------------------------
        % constructor
        %------------------------------------------------------------------
        function objects = energy_rx( varargin )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % return if no input argument
            if nargin == 0
                return;
            end

            % iterate arguments
            for index_arg = 2:numel( varargin )
                % multiple varargin{ 1 } / single varargin{ index_arg }
                if ~isscalar( varargin{ 1 } ) && isscalar( varargin{ index_arg } )
                    varargin{ index_arg } = repmat( varargin{ index_arg }, size( varargin{ 1 } ) );
                end
            end

            % ensure equal number of dimensions and sizes
            auxiliary.mustBeEqualSize( varargin{ : } );

            %--------------------------------------------------------------
            % 2.) create options
            %--------------------------------------------------------------
            % repeat default options
            objects = repmat( objects, size( varargin{ 1 } ) );

            % iterate options
            for index_object = 1:numel( objects )

                args = cell( 1, nargin );
                for index_arg = 1:nargin
                    args{ index_arg } = varargin{ index_arg }( index_object );
                end

                % set independent properties
                objects( index_object ) = set_properties( objects( index_object ), args{ : } );

            end % for index_object = 1:numel( objects )

        end % function objects = energy_rx( varargin )

        %------------------------------------------------------------------
        % set independent properties
        %------------------------------------------------------------------
        function energy_rx = set_properties( energy_rx, varargin )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % ensure class regularization.options.energy_rx (scalar)
            if ~( isa( energy_rx, 'regularization.options.energy_rx' ) && isscalar( energy_rx ) )
                errorStruct.message = 'energy_rx must be regularization.options.energy_rx!';
                errorStruct.identifier = 'set_properties:NoOptionsEnergy';
                error( errorStruct );
            end

            %--------------------------------------------------------------
            % 2.) set properties
            %--------------------------------------------------------------
            % iterate arguments
            for index_arg = 1:numel( varargin )

                if isa( varargin{ index_arg }, 'scattering.options.momentary' )

                    %--------------------------------------------------
                    % a) momentary scattering options
                    %------------------------------------------------------
                    energy_rx.momentary = varargin{ index_arg };

                elseif isa( varargin{ index_arg }, 'regularization.tgc.tgc' )

                    %------------------------------------------------------
                    % b) TGC options
                    %------------------------------------------------------
                    energy_rx.tgc = varargin{ index_arg };

                elseif isa( varargin{ index_arg }, 'regularization.dictionaries.dictionary' )

                    %------------------------------------------------------
                    % c) dictionary options
                    %------------------------------------------------------
                    energy_rx.dictionary = varargin{ index_arg };

                else

%                     %------------------------------------------------------
%                     % d) unknown class
%                     %------------------------------------------------------
%                     errorStruct.message = sprintf( 'Class of varargin{ %d } is unknown!', index_arg );
%                     errorStruct.identifier = 'energy_rx:UnknownClass';
%                     error( errorStruct );

                end % if isa( varargin{ index_arg }, 'scattering.options.momentary' )

            end % for index_arg = 1:numel( varargin )

        end % function energy_rx = set_properties( energy_rx, varargin )

        %------------------------------------------------------------------
        % create configurations
        %------------------------------------------------------------------
        function [ operators, LTs_dict, LTs_tgc, LTs_tgc_measurement ] = get_configs( options, operators )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % ensure two arguments
            narginchk( 2, 2 );

            % ensure class regularization.options.energy_rx
            if ~isa( options, 'regularization.options.energy_rx' )
                errorStruct.message = 'options must be regularization.options.energy_rx!';
                errorStruct.identifier = 'get_configs:NoCommonOptions';
                error( errorStruct );
            end

            % ensure class scattering.operator
            if ~isa( operators, 'scattering.operator' )
                errorStruct.message = 'operators must be scattering.operator!';
                errorStruct.identifier = 'get_configs:NoScatteringOperators';
                error( errorStruct );
            end

            % ensure equal number of dimensions and sizes
            [ options, operators ] = auxiliary.ensureEqualSize( options, operators );

            %--------------------------------------------------------------
            % 2.) create configurations
            %--------------------------------------------------------------
            % set momentary scattering operator options
            operators = set_options_momentary( operators, reshape( [ options.momentary ], size( options ) ) );

            % specify cell arrays
            LTs_dict = cell( size( options ) );
            LTs_tgc = cell( size( options ) );
            LTs_tgc_measurement = cell( size( options ) );

            % iterate ennergy options
            for index_options = 1:numel( options )

                %----------------------------------------------------------
                % a) create dictionary
                %----------------------------------------------------------
                LTs_dict{ index_options } = get_LTs( options( index_options ).dictionary, operators( index_options ) );

                %----------------------------------------------------------
                % b) time gain compensation (TGC)
                %----------------------------------------------------------
                [ LTs_tgc{ index_options }, LTs_tgc_measurement{ index_options } ] = get_LTs( options( index_options ).tgc, operators( index_options ) );

            end % for index_options = 1:numel( options )

            % convert cell arrays to arrays
            LTs_tgc = reshape( cat( 1, LTs_tgc{ : } ), size( options ) );

            % avoid cell arrays for single options
            if isscalar( options )
                LTs_dict = LTs_dict{ 1 };
                LTs_tgc_measurement = LTs_tgc_measurement{ 1 };
            end

        end % function [ operators, LTs_dict, LTs_tgc, LTs_tgc_measurement ] = get_configs( options, operators )

        %------------------------------------------------------------------
        % display options
        %------------------------------------------------------------------
        function show( options )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % ensure nonempty options
            if nargin <= 0 || isempty( options )
                options = regularization.options.energy_rx;
            end

            % ensure class regularization.options.energy_rx
            if ~isa( options, 'regularization.options.energy_rx' )
                errorStruct.message = 'options must be regularization.options.energy_rx!';
                errorStruct.identifier = 'show:NoOptions';
                error( errorStruct );
            end

            %--------------------------------------------------------------
            % 2.) display options
            %--------------------------------------------------------------
            % iterate ennergy options
            for index_object = 1:numel( options )

                %----------------------------------------------------------
                % print header
                %----------------------------------------------------------
                str_date_time = sprintf( '%04d-%02d-%02d: %02d:%02d:%02d', fix( clock ) );
                fprintf( ' %s\n', repmat( '-', [ 1, 80 ] ) );
                fprintf( ' %s (%s)\n', 'ennergy options', str_date_time );
                fprintf( ' %s\n', repmat( '-', [ 1, 80 ] ) );

                %----------------------------------------------------------
                % print content
                %----------------------------------------------------------
                fprintf( ' %-4s: %-13s %4s %-11s: %-5s %4s %-14s: %-13s\n', 'TGC', string( options( index_object ).tgc ), '', 'dictionary', string( options( index_object ).dictionary ), '', 'normalization', string( options( index_object ).normalization ) );

            end % for index_object = 1:numel( options )

        end % function show( options )

	end % methods

end % classdef energy_rx
