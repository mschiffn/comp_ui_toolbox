%
% superclass for all common regularization options
%
% author: Martin F. Schiffner
% date: 2019-12-28
% modified: 2020-01-03
%
classdef common

	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	%% properties
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	properties (SetAccess = private)

        % independent properties
        momentary ( 1, 1 ) scattering.options.momentary { mustBeNonempty } = scattering.options.momentary                       % momentary scattering options
        tgc ( 1, 1 ) regularization.options.tgc { mustBeNonempty } = regularization.options.tgc_off                                 % TGC options
        dictionary ( 1, 1 ) regularization.options.dictionary { mustBeNonempty } = regularization.options.dictionary_identity       % dictionary options
        normalization ( 1, 1 ) regularization.options.normalization { mustBeNonempty } = regularization.options.normalization_off	% normalization options

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

                % iterate arguments
                for index_arg = 1:numel( varargin )

                    if isa( varargin{ index_arg }, 'scattering.options.momentary' )

                        %--------------------------------------------------
                        % a) momentary scattering options
                        %--------------------------------------------------
                        objects( index_object ).momentary = varargin{ index_arg }( index_object );

                    elseif isa( varargin{ index_arg }, 'regularization.options.tgc' )

                        %--------------------------------------------------
                        % b) TGC options
                        %--------------------------------------------------
                        objects( index_object ).tgc = varargin{ index_arg }( index_object );

                    elseif isa( varargin{ index_arg }, 'regularization.options.dictionary' )

                        %--------------------------------------------------
                        % c) dictionary options
                        %--------------------------------------------------
                        objects( index_object ).dictionary = varargin{ index_arg }( index_object );

                    elseif isa( varargin{ index_arg }, 'regularization.options.normalization' )

                        %--------------------------------------------------
                        % d) normalization options
                        %--------------------------------------------------
                        objects( index_object ).normalization = varargin{ index_arg }( index_object );

                    else

                        %--------------------------------------------------
                        % e) unknown class
                        %--------------------------------------------------
                        errorStruct.message = sprintf( 'Class of varargin{ %d } is unknown!', index_arg );
                        errorStruct.identifier = 'common:UnknownClass';
                        error( errorStruct );

                    end % if isa( varargin{ index_arg }, 'scattering.options.momentary' )

                end % for index_arg = 1:numel( varargin )

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
                fprintf( ' %s\n', repmat( '=', [ 1, 80 ] ) );
                fprintf( ' %s (%s)\n', 'common regularization options', str_date_time );
                fprintf( ' %s\n', repmat( '=', [ 1, 80 ] ) );

                %----------------------------------------------------------
                % print content
                %----------------------------------------------------------
                fprintf( ' %-12s: %-13s %4s %-12s: %-5s %4s %-12s: %-13s\n', 'algorithm', show( options( index_object ).algorithm ), '', 'reweighting', show( options( index_object ).reweighting ), '', 'warm start', show( options( index_object ).warm_start ) );

            end % for index_object = 1:numel( options )

        end % function show( options )

	end % methods

end % classdef common
