%
% superclass for all numbers of nonzero components
%
% author: Martin F. Schiffner
% date: 2020-03-10
% modified: 2020-03-13
%
classdef NNZC < processing.metrics.region

	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	%% methods
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	methods

        %------------------------------------------------------------------
        % constructor
        %------------------------------------------------------------------
        function objects = NNZC( ROIs, boundaries_dB )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % superclass ensures class math.orthotope for ROIs
            % superclass ensures valid boundaries_dB

            %--------------------------------------------------------------
            % 2.) create numbers of nonzero components
            %--------------------------------------------------------------
            % constructor of superclass
            objects@processing.metrics.region( ROIs, boundaries_dB );

        end % function objects = NNZC( ROIs, boundaries_dB )

	end % methods

	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	%% methods (protected and hidden)
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	methods (Access = protected, Hidden)

        %------------------------------------------------------------------
        % evaluate samples (scalar)
        %------------------------------------------------------------------
        function result = evaluate_samples( ~, ~, indicator )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % calling function ensures class processing.metrics.volume (scalar) for volume
            % calling function ensures class math.grid_regular_orthogonal (scalar) for grid

            %--------------------------------------------------------------
            % 2.) compute volume
            %--------------------------------------------------------------
            result = sum( indicator( : ) );

        end % function result = evaluate_samples( ~, ~, indicator )

	end % methods (Access = protected, Hidden)

end % classdef NNZC < processing.metrics.region
