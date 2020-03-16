%
% superclass for all boxcar spatial anti-aliasing filters
%
% author: Martin F. Schiffner
% date: 2019-07-30
% modified: 2020-03-16
%
classdef boxcar < scattering.anti_aliasing_filters.on

	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	%% methods
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	methods

        %------------------------------------------------------------------
        % constructor
        %------------------------------------------------------------------
        function objects = boxcar( size )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % ensure nonempty size
            if nargin < 1 || isempty( size )
                size = 1;
            end

            % superclass ensures row vector for size
            % superclass ensures nonempty positive integers for size

            %--------------------------------------------------------------
            % 2.) create boxcar spatial anti-aliasing filters
            %--------------------------------------------------------------
            % constructor of superclass
            objects@scattering.anti_aliasing_filters.on( size );

        end % function objects = boxcar( size )

        %------------------------------------------------------------------
        % string array (implement string method)
        %------------------------------------------------------------------
        function strs_out = string( filters )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % ensure class scattering.anti_aliasing_filters.boxcar
            if ~isa( filters, 'scattering.anti_aliasing_filters.boxcar' )
                errorStruct.message = 'filters must be scattering.anti_aliasing_filters.boxcar!';
                errorStruct.identifier = 'string:NoBoxcarSpatialAntiAliasingFilters';
                error( errorStruct );
            end

            %--------------------------------------------------------------
            % 2.) create string array
            %--------------------------------------------------------------
            % repeat string "boxcar"
            strs_out = repmat( "boxcar", size( filters ) );

        end % function strs_out = string( filters )

	end % methods

	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	%% methods (protected and hidden)
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	methods (Access = protected, Hidden)

        %------------------------------------------------------------------
        % compute filter samples (scalar)
        %------------------------------------------------------------------
        function samples = compute_samples_scalar( ~, flags )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % calling method ensures class scattering.anti_aliasing_filters.anti_aliasing_filter for filter (scalar)
            % calling method ensures valid flags

            %--------------------------------------------------------------
            % 2.) compute filter samples (scalar)
            %--------------------------------------------------------------
            % detect valid grid points
            samples = double( all( flags < pi, 3 ) );

        end % function samples = compute_samples_scalar( ~, flags )

	end % methods (Access = protected, Hidden)

end % classdef boxcar < scattering.anti_aliasing_filters.on
