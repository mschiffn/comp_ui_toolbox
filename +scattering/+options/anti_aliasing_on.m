%
% abstract superclass for all active spatial anti-aliasing filters
%
% author: Martin F. Schiffner
% date: 2020-02-20
% modified: 2020-02-20
%
classdef (Abstract) anti_aliasing_on < scattering.options.anti_aliasing

	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	%% methods
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	methods

        %------------------------------------------------------------------
        % constructor
        %------------------------------------------------------------------
        function objects = anti_aliasing_on( varargin )

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

        end % function objects = anti_aliasing_on( varargin )

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
            % calling method ensures class scattering.options.anti_aliasing for filter (scalar)
            % calling method ensures class scattering.sequences.setups.setup for setup (scalar)
            % calling method ensures class scattering.sequences.setups.transducers.array_planar_regular_orthogonal for setup.xdc_array (scalar)
            % calling method ensures class processing.field for h_transfer (scalar)
            % calling method ensures ensure nonempty indices_element

            %--------------------------------------------------------------
            % 2.) apply spatial anti-aliasing filter (scalar)
            %--------------------------------------------------------------
            % compute flag reflecting the local angular spatial frequencies
            flag = compute_flag( setup, h_transfer, index_element );

            % compute samples of spatial anti-aliasing filter
            filter_samples = compute_samples_scalar( filter, flag );

            % apply anti-aliasing filter
            h_transfer = h_transfer .* processing.field( h_transfer.axis, h_transfer.grid_FOV, filter_samples );

        end % function h_transfer = apply_scalar( filter, setup, h_transfer, index_element )

	end % methods (Access = protected, Hidden)

	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	%% methods (Abstract, protected, hidden)
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	methods (Abstract, Access = protected, Hidden)

        %------------------------------------------------------------------
        % compute samples of spatial anti-aliasing filter (scalar)
        %------------------------------------------------------------------
        filter_samples = compute_samples_scalar( filter, flag )

	end % methods (Abstract, Access = protected, Hidden)

end % classdef (Abstract) anti_aliasing_on < scattering.options.anti_aliasing
