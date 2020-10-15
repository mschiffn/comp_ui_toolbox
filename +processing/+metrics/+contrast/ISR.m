%
% superclass for all integrated sidelobe ratios (ISRs)
%
% author: Martin F. Schiffner
% date: 2020-10-14
% modified: 2020-10-14
%
classdef ISR < processing.metrics.contrast.contrast

	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	%% methods
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	methods

        %------------------------------------------------------------------
        % constructor
        %------------------------------------------------------------------
        function objects = ISR( ROIs_main, ROIs_both, dynamic_ranges_dB )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % superclass validation functions ensure class math.orthotope for ROIs_main and ROIs_both

            % ensure equal subclasses of physical_values.length
%             auxiliary.mustBeEqualSubclasses( 'physical_values.length', ROIs_main.intervals.lb );

            %--------------------------------------------------------------
            % 2.) create integrated sidelobe ratios (ISRs)
            %--------------------------------------------------------------
            % constructor of superclass
            objects@processing.metrics.contrast.contrast( ROIs_main, ROIs_both, dynamic_ranges_dB );

        end % function objects = ISR( ROIs_main, ROIs_both, dynamic_ranges_dB )

	end % methods

	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	%% methods (protected and hidden)
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	methods (Access = protected, Hidden)

        %------------------------------------------------------------------
        % evaluate samples (scalar)
        %------------------------------------------------------------------
        function result = evaluate_samples( ISR, samples_1, samples_2 )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % calling function ensures class processing.metrics.contrast.ISR (scalar) for ISR

            %--------------------------------------------------------------
            % 2.) compute integrated sidelobe ratio (ISR)
            %--------------------------------------------------------------
            % compute histograms of samples_1 and samples_2
            samples_1_hist = histcounts( samples_1, (-ISR.dynamic_range_dB:0) );
            samples_2_hist = histcounts( samples_2, (-ISR.dynamic_range_dB:0) );

            % remove first histogram from second histogram
            samples_hist = samples_2_hist - samples_1_hist;

            % compute expectations
            result = 1 - ( max( samples_1 ) - samples_hist * ( 0.5 + (-ISR.dynamic_range_dB + 1:0) )' / sum( samples_hist ) ) / ISR.dynamic_range_dB;

        end % function result = evaluate_samples( ISR, samples_1, samples_2 )

	end % methods (Access = protected, Hidden)

end % classdef ISR < processing.metrics.contrast.contrast
