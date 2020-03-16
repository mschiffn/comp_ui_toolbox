%
% abstract superclass for all active spatial anti-aliasing filters
%
% author: Martin F. Schiffner
% date: 2020-02-20
% modified: 2020-03-09
%
classdef (Abstract) on < scattering.anti_aliasing_filters.anti_aliasing_filter

	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	%% methods
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	methods

        %------------------------------------------------------------------
        % constructor
        %------------------------------------------------------------------
        function objects = on( size )

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
            % 2.) create active spatial anti-aliasing filters
            %--------------------------------------------------------------
            % constructor of superclass
            objects@scattering.anti_aliasing_filters.anti_aliasing_filter( size );

        end % function objects = on( size )

	end % methods

	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	%% methods (protected and hidden)
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	methods (Access = protected, Hidden)

        %------------------------------------------------------------------
        % apply spatial anti-aliasing filter (scalar)
        %------------------------------------------------------------------
        function h_transfer = apply_scalar( filter, setup, h_transfer, index_element )

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
            % compute flags reflecting the local angular spatial frequencies
            flags = compute_flags( setup, h_transfer.axis, index_element );

            % compute filter samples (scalar)
            filter_samples = compute_samples_scalar( filter, flags.samples );

            % apply anti-aliasing filter
            h_transfer = h_transfer .* processing.field( h_transfer.axis, h_transfer.grid_FOV, filter_samples );

        end % function h_transfer = apply_scalar( filter, setup, h_transfer, index_element )

	end % methods (Access = protected, Hidden)

end % classdef (Abstract) on < scattering.anti_aliasing_filters.anti_aliasing_filter
