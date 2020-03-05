%
% superclass for all boxcar spatial anti-aliasing filters
%
% author: Martin F. Schiffner
% date: 2019-07-30
% modified: 2020-03-04
%
classdef boxcar < scattering.anti_aliasing_filters.on

	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	%% methods
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	methods

        %------------------------------------------------------------------
        % constructor
        %------------------------------------------------------------------
        function objects = boxcar( varargin )

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
            % 2.) create boxcar spatial anti-aliasing filter options
            %--------------------------------------------------------------
            % constructor of superclass
            objects@scattering.anti_aliasing_filters.on( size );

        end % function objects = boxcar( varargin )

        %------------------------------------------------------------------
        % compute spatial anti-aliasing filters
        %------------------------------------------------------------------
        function filters = compute_filter( options_anti_aliasing, flags )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % ensure class scattering.anti_aliasing_filters.boxcar
            if ~isa( options_anti_aliasing, 'scattering.anti_aliasing_filters.boxcar' )
                errorStruct.message = 'options_anti_aliasing must be scattering.anti_aliasing_filters.boxcar!';
                errorStruct.identifier = 'compute_filter:NoOptionsAntiAliasingBoxcar';
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
                filters{ index_filter } = all( flags{ index_filter } < pi, 3 );

            end % for index_filter = 1:numel( options_anti_aliasing )

            % avoid cell array for single options_anti_aliasing
            if isscalar( options_anti_aliasing )
                filters = filters{ 1 };
            end

        end % function filters = compute_filter( options_anti_aliasing, flags )

        %------------------------------------------------------------------
        % string array (implement string method)
        %------------------------------------------------------------------
        function strs_out = string( anti_aliasings_boxcar )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % ensure class scattering.anti_aliasing_filters.boxcar
            if ~isa( anti_aliasings_boxcar, 'scattering.anti_aliasing_filters.boxcar' )
                errorStruct.message = 'anti_aliasings_boxcar must be scattering.anti_aliasing_filters.boxcar!';
                errorStruct.identifier = 'string:NoOptionsAntiAliasingBoxcar';
                error( errorStruct );
            end

            %--------------------------------------------------------------
            % 2.) create string array
            %--------------------------------------------------------------
            % repeat string "boxcar"
            strs_out = repmat( "boxcar", size( anti_aliasings_boxcar ) );

        end % function strs_out = string( anti_aliasings_boxcar )

	end % methods

	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	%% methods (protected and hidden)
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	methods (Access = protected, Hidden)

        %------------------------------------------------------------------
        % compute samples of spatial anti-aliasing filter (scalar)
        %------------------------------------------------------------------
        function filter_samples = compute_samples_scalar( ~, flag )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % calling method ensures class scattering.anti_aliasing_filters.anti_aliasing_filter for filter (scalar)
            % calling method ensures valid flag

            %--------------------------------------------------------------
            % 2.) apply spatial anti-aliasing filter (scalar)
            %--------------------------------------------------------------
            % detect valid grid points
            filter_samples = all( flag < pi, 3 );

        end % function filter_samples = compute_samples_scalar( ~, flag )

	end % methods (Access = protected, Hidden)

end % classdef boxcar < scattering.anti_aliasing_filters.on
