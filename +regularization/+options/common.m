%
% superclass for all common regularization options
%
% author: Martin F. Schiffner
% date: 2019-12-28
% modified: 2020-02-17
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

	end % methods

end % classdef common
