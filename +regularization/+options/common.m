%
% superclass for all common regularization options
%
% author: Martin F. Schiffner
% date: 2019-12-28
% modified: 2020-02-26
%
classdef common < regularization.options.energy_rx

	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	%% properties
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	properties (SetAccess = private)

        % independent properties
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
        function objects = common( options_energy, normalizations )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % ensure class regularization.options.energy_rx
            if ~isa( options_energy, 'regularization.options.energy_rx' )
                errorStruct.message = 'options_energy must be regularization.options.energy_rx!';
                errorStruct.identifier = 'common:NoEnergyOptions';
                error( errorStruct );
            end

            % ensure class regularization.normalizations.normalization
            if ~isa( normalizations, 'regularization.normalizations.normalization' )
                errorStruct.message = 'normalizations must be regularization.normalizations.normalization!';
                errorStruct.identifier = 'common:NoNormalizations';
                error( errorStruct );
            end

            % multiple options_energy / single normalizations
            if ~isscalar( options_energy ) && isscalar( normalizations )
                normalizations = repmat( normalizations, size( options_energy ) );
            end

            % single options_energy / multiple normalizations
            if isscalar( options_energy ) && ~isscalar( normalizations )
                options_energy = repmat( options_energy, size( normalizations ) );
            end

            % ensure equal number of dimensions and sizes
            auxiliary.mustBeEqualSize( options_energy, normalizations );

            %--------------------------------------------------------------
            % 2.) create options
            %--------------------------------------------------------------
            % constructor of superclass
            objects@regularization.options.energy_rx( [ options_energy.momentary ], [ options_energy.tgc ], [ options_energy.dictionary ] );

            % reshape common options
            objects = reshape( objects, size( options_energy ) );

            % iterate options
            for index_object = 1:numel( objects )

                % set independent properties
                objects( index_object ).normalization = normalizations( index_object );

            end % for index_object = 1:numel( objects )

        end % function objects = common( options_energy, normalizations )

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
            % call set_properties method of superclass
            common = set_properties@regularization.options.energy_rx( common, varargin{ : } );

            % iterate arguments
            for index_arg = 1:numel( varargin )

                if isa( varargin{ index_arg }, 'regularization.normalizations.normalization' )

                    %------------------------------------------------------
                    % a) normalization options
                    %------------------------------------------------------
                    common.normalization = varargin{ index_arg };

                else
% 
%                     %------------------------------------------------------
%                     % b) unknown class
%                     %------------------------------------------------------
%                     errorStruct.message = sprintf( 'Class of varargin{ %d } is unknown!', index_arg );
%                     errorStruct.identifier = 'common:UnknownClass';
%                     error( errorStruct );

                end % if isa( varargin{ index_arg }, 'regularization.normalizations.normalization' )

            end % for index_arg = 1:numel( varargin )

        end % function common = set_properties( common, varargin )

        %------------------------------------------------------------------
        % create configurations
        %------------------------------------------------------------------
        function [ operators, LTs_dict, LTs_tgc, LTs_tgc_measurement ] = get_configs( options, operators )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % ensure class regularization.options.common
            if ~isa( options, 'regularization.options.common' )
                errorStruct.message = 'options must be regularization.options.common!';
                errorStruct.identifier = 'get_configs:NoCommonOptions';
                error( errorStruct );
            end

            % ensure class scattering.operator
            if ~isa( operators, 'scattering.operator' )
                errorStruct.message = 'operators must be scattering.operator!';
                errorStruct.identifier = 'get_configs:NoScatteringOperators';
                error( errorStruct );
            end

            % multiple options / single operators
            if ~isscalar( options ) && isscalar( operators )
                operators = repmat( operators, size( options ) );
            end

            % single options / multiple operators
            if isscalar( options ) && ~isscalar( operators )
                options = repmat( options, size( operators ) );
            end

            % ensure equal number of dimensions and sizes
            auxiliary.mustBeEqualSize( options, operators );

            %--------------------------------------------------------------
            % 2.) create configurations
            %--------------------------------------------------------------
            % call get_configs method of superclass
            [ operators, LTs_dict, LTs_tgc, LTs_tgc_measurement ] = get_configs@regularization.options.energy_rx( options, operators );

            % ensure cell arrays
            if ~iscell( LTs_dict )
                LTs_dict = { LTs_dict };
                LTs_tgc_measurement = { LTs_tgc_measurement };
            end

            % iterate common regularization options
            for index_options = 1:numel( options )

                %----------------------------------------------------------
                % a) apply normalization
                %----------------------------------------------------------
% TODO: LTs_dict{ index_options } = get_LTs( options( index_options ).normalization, operators( index_options ), LTs_dict{ index_options } );
                if ~isa( options( index_options ).normalization, 'regularization.normalizations.off' )

                    % compute received energies
                    E_M = energy_rx_scalar( operators( index_options ), LTs_dict{ index_options }, LTs_tgc_measurement{ index_options } );

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

                % print header
                auxiliary.print_header( "common regularization options", '-' );

                % print content
                fprintf( ' %-4s: %-13s %4s %-11s: %-5s %4s %-14s: %-13s\n', 'TGC', string( options( index_object ).tgc ), '', 'dictionary', string( options( index_object ).dictionary ), '', 'normalization', string( options( index_object ).normalization ) );

            end % for index_object = 1:numel( options )

        end % function show( options )

	end % methods

end % classdef common < regularization.options.energy_rx
