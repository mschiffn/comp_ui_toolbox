%
% superclass for all volumes
%
% author: Martin F. Schiffner
% date: 2020-03-13
% modified: 2020-10-14
%
classdef volume < processing.metrics.region.region

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
            % ensure one or two arguments
            narginchk( 1, 2 );

            % superclass ensures class scattering.sequences.setups.geometry.shape for ROIs

            % ensure definition of boundaries_dB
            if nargin < 2
                boundaries_dB = [];
            end

            % superclass ensures nonempty negative double for boundaries_dB

            %--------------------------------------------------------------
            % 2.) create numbers of nonzero components
            %--------------------------------------------------------------
            % constructor of superclass
            objects@processing.metrics.region.region( ROIs, boundaries_dB );

        end % function objects = volume( ROIs, boundaries_dB )

	end % methods

	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	%% methods (protected and hidden)
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	methods (Access = protected, Hidden)

        %------------------------------------------------------------------
        % evaluate metric (samples)
        %------------------------------------------------------------------
        function result = evaluate_samples( ~, delta_V, indicator )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % calling function ensures class processing.metrics.region.region (scalar) for region
            % calling function ensures volume element for delta_V
            % calling function ensures logical for indicator

            %--------------------------------------------------------------
            % 2.) compute volume
            %--------------------------------------------------------------
            result = sum( indicator( : ) ) * delta_V;

        end % function result = evaluate_samples( ~, delta_V, indicator )

	end % methods (Access = protected, Hidden)

end % classdef volume < processing.metrics.region.region
