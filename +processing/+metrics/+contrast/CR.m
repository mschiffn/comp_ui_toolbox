%
% superclass for all contrast ratios (CRs)
%
% author: Martin F. Schiffner
% date: 2020-03-14
% modified: 2020-10-14
%
classdef CR < processing.metrics.contrast.contrast

	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	%% methods
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	methods

        %------------------------------------------------------------------
        % constructor
        %------------------------------------------------------------------
        function objects = CR( ROIs_1, ROIs_2, dynamic_ranges_dB )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % superclass validation functions ensure class math.orthotope for ROIs_ref and ROIs_noise

            % ensure equal subclasses of physical_values.length
%             auxiliary.mustBeEqualSubclasses( 'physical_values.length', ROIs_ref.intervals.lb );

            %--------------------------------------------------------------
            % 2.) create contrast ratios (CRs)
            %--------------------------------------------------------------
            % constructor of superclass
            objects@processing.metrics.contrast.contrast( ROIs_1, ROIs_2, dynamic_ranges_dB );

        end % function objects = CR( ROIs_1, ROIs_2, dynamic_ranges_dB )

	end % methods

	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	%% methods (protected and hidden)
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	methods (Access = protected, Hidden)

        %------------------------------------------------------------------
        % evaluate samples (scalar)
        %------------------------------------------------------------------
        function result = evaluate_samples( ~, samples_1, samples_2 )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % calling function ensures class processing.metrics.CR (scalar) for CR

            %--------------------------------------------------------------
            % 2.) compute contrast ratio (CR)
            %--------------------------------------------------------------
            % means
            samples_1_mean = mean( samples_1 );
            samples_2_mean = mean( samples_2 );

            % contrast-to-noise ratio (CNR)
            result = abs( samples_1_mean - samples_2_mean );

        end % function result = evaluate_samples( ~, samples_1, samples_2 )

	end % methods (Access = protected, Hidden)

end % classdef CR < processing.metrics.contrast.contrast
