%
% superclass for all generalized contrast-to-noise ratios (gCNRs)
%
% author: Martin F. Schiffner
% date: 2020-02-29
% modified: 2020-10-14
%
classdef gCNR < processing.metrics.contrast.contrast

	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	%% methods
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	methods

        %------------------------------------------------------------------
        % constructor
        %------------------------------------------------------------------
        function objects = gCNR( ROIs_1, ROIs_2, dynamic_ranges_dB )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % superclass validation functions ensure class math.orthotope for ROIs_1 and ROIs_2

            % ensure equal subclasses of physical_values.length
%             auxiliary.mustBeEqualSubclasses( 'physical_values.length', ROIs_1.intervals.lb );

            %--------------------------------------------------------------
            % 2.) create generalized contrast-to-noise ratios (gCNRs)
            %--------------------------------------------------------------
            % constructor of superclass
            objects@processing.metrics.contrast.contrast( ROIs_1, ROIs_2, dynamic_ranges_dB );

        end % function objects = gCNR( ROIs_1, ROIs_2, dynamic_ranges_dB )

	end % methods

	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	%% methods (protected and hidden)
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	methods (Access = protected, Hidden)

        %------------------------------------------------------------------
        % evaluate samples (scalar)
        %------------------------------------------------------------------
        function result = evaluate_samples( gCNR, samples_1, samples_2 )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % calling function ensures class processing.metrics.gCNR (scalar) for gCNR

            %--------------------------------------------------------------
            % 2.) compute generalized contrast-to-noise ratio (gCNR)
            %--------------------------------------------------------------
            % estimate PDFs of samples_1 and samples_2
            samples_1_pdf = histcounts( samples_1, (-gCNR.dynamic_range_dB:0), 'Normalization', 'pdf' );
            samples_2_pdf = histcounts( samples_2, (-gCNR.dynamic_range_dB:0), 'Normalization', 'pdf' );

            % overlap of PDFs and gCNR
            overlap = sum( min( samples_1_pdf, samples_2_pdf ) );
            result = 1 - overlap;

        end % function result = evaluate_samples( gCNR, samples_1, samples_2 )

	end % methods (Access = protected, Hidden)

end % classdef gCNR < processing.metrics.contrast.contrast
