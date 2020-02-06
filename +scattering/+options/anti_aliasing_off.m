%
% superclass for all inactive spatial anti-aliasing filter options
%
% author: Martin F. Schiffner
% date: 2019-07-29
% modified: 2020-02-01
%
classdef anti_aliasing_off < scattering.options.anti_aliasing

	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	%% methods
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	methods

        %------------------------------------------------------------------
        % constructor
        %------------------------------------------------------------------
        function objects = anti_aliasing_off( varargin )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % ensure nonempty size
            if nargin >= 1 && ~isempty( varargin{ 1 } )
                size = varargin{ 1 };
            else
                size = 1;
            end

            % superclass ensures row vectors for size
            % superclass ensures positive integers for size

            %--------------------------------------------------------------
            % 2.) create inactive spatial anti-aliasing filter options
            %--------------------------------------------------------------
            % constructor of superclass
            objects@scattering.options.anti_aliasing( size );

        end % function objects = anti_aliasing_off( varargin )

        %------------------------------------------------------------------
        % compute spatial anti-aliasing filters
        %------------------------------------------------------------------
        function filters = compute_filter( options_anti_aliasing, flags )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % ensure class scattering.options.anti_aliasing_off
            if ~isa( options_anti_aliasing, 'scattering.options.anti_aliasing_off' )
                errorStruct.message = 'options_anti_aliasing must be scattering.options.anti_aliasing_off!';
                errorStruct.identifier = 'compute_filter:NoOptionsAntiAliasingOff';
                error( errorStruct );
            end

            % ensure cell array for flags
            if ~iscell( flags )
                flags = { flags };
            end

            % ensure equal number of dimensions and sizes
            auxiliary.mustBeEqualSize( options_anti_aliasing, flags );

            %--------------------------------------------------------------
            % 2.) compute spatial anti-aliasing filters
            %--------------------------------------------------------------
            % specify cell array for filters
            filters = cell( size( options_anti_aliasing ) );

            % iterate spatial anti-aliasing filter options
            for index_filter = 1:numel( options_anti_aliasing )

                % detect valid grid points
                filters{ index_filter } = true( size( flags{ index_filter } ) );

            end % for index_filter = 1:numel( options_anti_aliasing )

            % avoid cell array for single options_anti_aliasing
            if isscalar( options_anti_aliasing )
                filters = filters{ 1 };
            end

        end % function filters = compute_filter( options_anti_aliasing, flags )

        %------------------------------------------------------------------
        % string array (overload string method)
        %------------------------------------------------------------------
        function strs_out = string( anti_aliasings_off )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % ensure class scattering.options.anti_aliasing_off
            if ~isa( anti_aliasings_off, 'scattering.options.anti_aliasing_off' )
                errorStruct.message = 'anti_aliasings_off must be scattering.options.anti_aliasing_off!';
                errorStruct.identifier = 'string:NoOptionsAlgorithmDirect';
                error( errorStruct );
            end

            %--------------------------------------------------------------
            % 2.) create string array
            %--------------------------------------------------------------
            % repeat string "off"
            strs_out = repmat( "off", size( anti_aliasings_off ) );

        end % function strs_out = string( anti_aliasings_off )

	end % methods

end % classdef anti_aliasing_off < scattering.options.anti_aliasing
