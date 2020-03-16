%
% superclass for all inactive spatial anti-aliasing filters
%
% author: Martin F. Schiffner
% date: 2019-07-29
% modified: 2020-03-09
%
classdef off < scattering.anti_aliasing_filters.anti_aliasing_filter

	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	%% methods
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	methods

        %------------------------------------------------------------------
        % constructor
        %------------------------------------------------------------------
        function objects = off( size )

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
            % 2.) create inactive spatial anti-aliasing filters
            %--------------------------------------------------------------
            % constructor of superclass
            objects@scattering.anti_aliasing_filters.anti_aliasing_filter( size );

        end % function objects = off( size )

        %------------------------------------------------------------------
        % string array (implement string method)
        %------------------------------------------------------------------
        function strs_out = string( filters_off )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % ensure class scattering.anti_aliasing_filters.off
            if ~isa( filters_off, 'scattering.anti_aliasing_filters.off' )
                errorStruct.message = 'filters_off must be scattering.anti_aliasing_filters.off!';
                errorStruct.identifier = 'string:NoInactiveSpatialAntiAliasingFilters';
                error( errorStruct );
            end

            %--------------------------------------------------------------
            % 2.) create string array
            %--------------------------------------------------------------
            % repeat string "off"
            strs_out = repmat( "off", size( filters_off ) );

        end % function strs_out = string( filters_off )

	end % methods

	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	%% methods (protected and hidden)
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	methods (Access = protected, Hidden)

        %------------------------------------------------------------------
        % apply spatial anti-aliasing filter (scalar)
        %------------------------------------------------------------------
        function h_transfer = apply_scalar( ~, ~, h_transfer, ~ )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % calling method ensures class scattering.anti_aliasing_filters.anti_aliasing_filter for filter (scalar)
            % calling method ensures class scattering.sequences.setups.setup for setup (scalar)
            % calling method ensures class scattering.sequences.setups.transducers.array_planar_regular_orthogonal for setup.xdc_array (scalar)
            % calling method ensures class processing.field for h_transfer (scalar)
            % calling method ensures ensure nonempty indices_element

            %--------------------------------------------------------------
            % 2.) apply spatial anti-aliasing filter (scalar)
            %--------------------------------------------------------------
            % copy spatial transfer function

        end % function h_transfer = apply_scalar( ~, ~, h_transfer, ~ )

        %------------------------------------------------------------------
        % compute filter samples (scalar)
        %------------------------------------------------------------------
        function coefficients = compute_samples_scalar( filter, flags )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % calling function ensures class scattering.anti_aliasing_filters.anti_aliasing_filter for filter

            % ensure class scattering.anti_aliasing_filters.off
            if ~isa( filter, 'scattering.anti_aliasing_filters.off' )
                errorStruct.message = 'filter must be scattering.anti_aliasing_filters.off!';
                errorStruct.identifier = 'compute_samples_scalar:NoInactiveSpatialAntiAliasingFilters';
                error( errorStruct );
            end

            % calling function ensures cell array for flags

            %--------------------------------------------------------------
            % 2.) compute filter samples (scalar)
            %--------------------------------------------------------------
            % all grid points are valid
            coefficients = ones( size( flags ) );

        end % function coefficients = compute_samples_scalar( filter, flags )

	end % methods (Access = protected, Hidden)

end % classdef off < scattering.anti_aliasing_filters.anti_aliasing_filter
