%
% superclass for all scattering operator options
%
% author: Martin F. Schiffner
% date: 2019-02-15
% modified: 2019-08-03
%
classdef options

	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	%% properties
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	properties (SetAccess = private)

        % independent properties
        static ( 1, 1 ) scattering.options.static = scattering.options.static           % static options
        momentary ( 1, 1 ) scattering.options.momentary = scattering.options.momentary	% momentary options

    end % properties

	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	%% methods
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	methods

        %------------------------------------------------------------------
        % constructor
        %------------------------------------------------------------------
        function objects = options( varargin )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % return default object for missing arguments
            if nargin == 0
                return;
            end

            % ensure nonempty static
            if ~isempty( varargin{ 1 } )
                static = varargin{ 1 };
            else
                static = scattering.options.static;
            end

            % ensure nonempty momentary
            if nargin >= 2 && ~isempty( varargin{ 2 } )
                momentary = varargin{ 2 };
            else
                momentary = scattering.options.momentary;
            end

            % multiple static / single momentary
            if ~isscalar( static ) && isscalar( momentary )
                momentary = repmat( momentary, size( static ) );
            end

            % single static / multiple momentary
            if isscalar( static ) && ~isscalar( momentary )
                static = repmat( static, size( momentary ) );
            end

            % ensure equal number of dimensions and sizes
            auxiliary.mustBeEqualSize( static, momentary );

            %--------------------------------------------------------------
            % 2.) create scattering operator options
            %--------------------------------------------------------------
            % repeat default scattering operator options
            objects = repmat( objects, size( static ) );

            % iterate scattering operator options
            for index_object = 1:numel( static )

                % set independent properties
                objects( index_object ).static = static( index_object );
                objects( index_object ).momentary = momentary( index_object );

            end % for index_object = 1:numel( static )

        end % function objects = options( varargin )

        %------------------------------------------------------------------
        % set properties of momentary scattering operator options
        %------------------------------------------------------------------
        function options = set_properties_momentary( options, varargin )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % ensure class scattering.options
            if ~isa( options, 'scattering.options' )
                errorStruct.message = 'options must be scattering.options!';
                errorStruct.identifier = 'set_properties_momentary:NoOptions';
                error( errorStruct );
            end

            % ensure equal number of dimensions and sizes
            auxiliary.mustBeEqualSize( options, varargin{ : } );

            %--------------------------------------------------------------
            % 2.) set momentary scattering operator options
            %--------------------------------------------------------------
            % specify cell array for arguments
            args = cell( size( varargin ) );

            % iterate scattering operators
            for index_object = 1:numel( options )

                % process arguments
                for index_arg = 1:numel( varargin )
                    args{ index_arg } = varargin{ index_arg }( index_object );
                end

                % set properties of momentary scattering operator options
                options( index_object ).momentary = set_properties( options( index_object ).momentary, args{ : } );

            end % for index_object = 1:numel( options )

        end % function options = set_properties_momentary( options, varargin )

	end % methods

end % classdef options
