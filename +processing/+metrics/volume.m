%
% superclass for all volumes
%
% author: Martin F. Schiffner
% date: 2020-03-13
% modified: 2020-03-13
%
classdef volume < processing.metrics.region

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	%% methods
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    methods

        %------------------------------------------------------------------
        % constructor
        %------------------------------------------------------------------
        function objects = volume( ROIs, boundaries_dB )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % superclass ensures class math.orthotope for ROIs
            % superclass ensures valid boundaries_dB

            %--------------------------------------------------------------
            % 2.) create volumes
            %--------------------------------------------------------------
            % constructor of superclass
            objects@processing.metrics.region( ROIs, boundaries_dB );

        end % function objects = volume( ROIs, boundaries_dB )

	end % methods

	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	%% methods (protected and hidden)
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	methods (Access = protected, Hidden)

        %------------------------------------------------------------------
        % evaluate samples (scalar)
        %------------------------------------------------------------------
        function result = evaluate_samples( ~, grid, indicator )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % calling function ensures class processing.metrics.volume (scalar) for volume
            % calling function ensures class math.grid_regular_orthogonal (scalar) for grid

            %--------------------------------------------------------------
            % 2.) compute volume
            %--------------------------------------------------------------
            N_samples = sum( indicator( : ) );
            result = N_samples * grid.cell_ref.volume;

        end % function result = evaluate_samples( ~, grid, indicator )

	end % methods (Access = protected, Hidden)

end % classdef volume < processing.metrics.region
