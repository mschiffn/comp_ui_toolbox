%
% superclass for all common regularization options
%
% author: Martin F. Schiffner
% date: 2019-12-28
% modified: 2020-02-18
%
classdef common

	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	%% properties
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	properties (SetAccess = private)

        % independent properties
        momentary ( 1, 1 ) scattering.options.momentary { mustBeNonempty } = scattering.options.momentary                           % momentary scattering options
        tgc ( 1, 1 ) regularization.tgc.tgc { mustBeNonempty } = regularization.tgc.off                                             % TGC
        dictionary ( 1, 1 ) regularization.dictionaries.dictionary { mustBeNonempty } = regularization.dictionaries.identity        % dictionary
        normalization ( 1, 1 ) regularization.normalizations.normalization { mustBeNonempty } = regularization.normalizations.off	% normalization options
        display ( 1, 1 ) logical { mustBeNonempty } = 1                                                                             % display results of estimate

	end % properties

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% methods
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	methods

        %------------------------------------------------------------------
        % constructor
        %------------------------------------------------------------------
        function objects = common( varargin )

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

        end % function objects = common( varargin )

        %------------------------------------------------------------------
        % display options
        %------------------------------------------------------------------
        function show( options )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % ensure nonempty options
            if nargin <= 0 || isempty( options )
                options = regularization.options.common;
            end

            % ensure class regularization.options.common
            if ~isa( options, 'regularization.options.common' )
                errorStruct.message = 'options must be regularization.options.common!';
                errorStruct.identifier = 'show:NoOptions';
                error( errorStruct );
            end

            %--------------------------------------------------------------
            % 2.) display options
            %--------------------------------------------------------------
            % iterate common regularization options
            for index_object = 1:numel( options )

                %----------------------------------------------------------
                % print header
                %----------------------------------------------------------
                str_date_time = sprintf( '%04d-%02d-%02d: %02d:%02d:%02d', fix( clock ) );
                fprintf( ' %s\n', repmat( '-', [ 1, 80 ] ) );
                fprintf( ' %s (%s)\n', 'common regularization options', str_date_time );
                fprintf( ' %s\n', repmat( '-', [ 1, 80 ] ) );

                %----------------------------------------------------------
                % print content
                %----------------------------------------------------------
                fprintf( ' %-4s: %-13s %4s %-11s: %-5s %4s %-14s: %-13s\n', 'TGC', string( options( index_object ).tgc ), '', 'dictionary', string( options( index_object ).dictionary ), '', 'normalization', string( options( index_object ).normalization ) );

            end % for index_object = 1:numel( options )

        end % function show( options )

        %------------------------------------------------------------------
        % set independent properties
        %------------------------------------------------------------------
        function common = set_properties( common, varargin )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % ensure class regularization.options.common (scalar)
            if ~( isa( common, 'regularization.options.common' ) && isscalar( common ) )
                errorStruct.message = 'common must be regularization.options.common!';
                errorStruct.identifier = 'set_properties:NoOptionsMomentary';
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
                    common.momentary = varargin{ index_arg };

                elseif isa( varargin{ index_arg }, 'regularization.tgc.tgc' )

                    %------------------------------------------------------
                    % b) TGC options
                    %------------------------------------------------------
                    common.tgc = varargin{ index_arg };

                elseif isa( varargin{ index_arg }, 'regularization.dictionaries.dictionary' )

                    %------------------------------------------------------
                    % c) dictionary options
                    %------------------------------------------------------
                    common.dictionary = varargin{ index_arg };

                elseif isa( varargin{ index_arg }, 'regularization.normalizations.normalization' )

                    %------------------------------------------------------
                    % d) normalization options
                    %------------------------------------------------------
                    common.normalization = varargin{ index_arg };

                else

                    %------------------------------------------------------
                    % e) unknown class
                    %------------------------------------------------------
                    errorStruct.message = sprintf( 'Class of varargin{ %d } is unknown!', index_arg );
                    errorStruct.identifier = 'common:UnknownClass';
                    error( errorStruct );

                end % if isa( varargin{ index_arg }, 'scattering.options.momentary' )

            end % for index_arg = 1:numel( varargin )

        end % function common = set_properties( common, varargin )

        %------------------------------------------------------------------
        % create configurations
        %------------------------------------------------------------------
        function [ operators_born, LTs_dict, LTs_tgc, LTs_tgc_measurement ] = get_configs( options, operators_born )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % ensure class regularization.options.common
            if ~isa( options, 'regularization.options.common' )
                errorStruct.message = 'options must be regularization.options.common!';
                errorStruct.identifier = 'get_configs:NoCommonOptions';
                error( errorStruct );
            end

            % ensure class scattering.operator_born
            if ~isa( operators_born, 'scattering.operator_born' )
                errorStruct.message = 'operators_born must be scattering.operator_born!';
                errorStruct.identifier = 'get_configs:NoOperatorsBorn';
                error( errorStruct );
            end

            % multiple options / single operators_born
            if ~isscalar( options ) && isscalar( operators_born )
                operators_born = repmat( operators_born, size( options ) );
            end

            % single options / multiple operators_born
            if isscalar( options ) && ~isscalar( operators_born )
                options = repmat( options, size( operators_born ) );
            end

            % ensure equal number of dimensions and sizes
            auxiliary.mustBeEqualSize( options, operators_born );

            %--------------------------------------------------------------
            % 2.) create configurations
            %--------------------------------------------------------------
            % set momentary scattering operator options
            operators_born = set_options_momentary( operators_born, reshape( [ options.momentary ], size( options ) ) );

            % specify cell arrays
            LTs_dict = cell( size( options ) );
            LTs_tgc = cell( size( options ) );
            LTs_tgc_measurement = cell( size( options ) );

            % iterate common regularization options
            for index_options = 1:numel( options )

                %----------------------------------------------------------
                % a) time gain compensation (TGC)
                %----------------------------------------------------------
                [ LTs_tgc{ index_options }, LTs_tgc_measurement{ index_options } ] = get_LTs( options( index_options ).tgc, operators_born( index_options ) );

                %----------------------------------------------------------
                % b) create dictionary
                %----------------------------------------------------------
                LTs_dict{ index_options } = get_LTs( options( index_options ).dictionary, operators_born( index_options ) );

                %----------------------------------------------------------
                % c) apply normalization
                %----------------------------------------------------------
                if ~isa( options( index_options ).normalization, 'regularization.normalizations.off' )

                    % compute received energies
                    E_M = energy_rx_scalar( operators_born( index_options ), LTs_dict{ index_options }, LTs_tgc_measurement{ index_options } );

                    % create inverse weighting matrix
                    LT_weighting_inv = linear_transforms.weighting( 1 ./ sqrt( double( E_M ) ) );

                    % apply normalization settings
                    LT_weighting_inv = apply( options( index_options ).normalization, LT_weighting_inv );

                    % composition with non-canonical linear transform
% TODO: neglect identity in composition
                    if ~isa( LTs_dict{ index_options }, 'linear_transforms.identity' )
                        LT_weighting_inv = linear_transforms.composition( LT_weighting_inv, LTs_dict{ index_options } );
                    end

                    % update dictionary
                    LTs_dict{ index_options } = LT_weighting_inv;

                end % if ~isa( options( index_options ).normalization, 'regularization.normalizations.off' )

            end % for index_options = 1:numel( options )

            % convert cell arrays to arrays
            LTs_tgc = reshape( cat( 1, LTs_tgc{ : } ), size( options ) );

            % avoid cell arrays for single options
            if isscalar( options )
                LTs_dict = LTs_dict{ 1 };
                LTs_tgc_measurement = LTs_tgc_measurement{ 1 };
            end

        end % function [ operators_born, LTs_dict, LTs_tgc, LTs_tgc_measurement ] = get_configs( options, operators_born )

	end % methods

end % classdef common
