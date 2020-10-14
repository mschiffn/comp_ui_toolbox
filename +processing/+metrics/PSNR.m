%
% superclass for all peak signal-to-noise ratios (PSNRs)
%
% author: Martin F. Schiffner
% date: 2020-10-13
% modified: 2020-10-13
%
classdef PSNR < processing.metrics.contrast

	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	%% methods
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	methods

        %------------------------------------------------------------------
        % constructor
        %------------------------------------------------------------------
        function objects = PSNR( ROIs_1, ROIs_2, dynamic_ranges_dB )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % superclass validation functions ensure class math.orthotope for ROIs_1 and ROIs_2

            % ensure equal subclasses of physical_values.length
%             auxiliary.mustBeEqualSubclasses( 'physical_values.length', ROIs_1.intervals.lb );

            %--------------------------------------------------------------
            % 2.) create peak signal-to-noise ratios (PSNRs)
            %--------------------------------------------------------------
            % constructor of superclass
            objects@processing.metrics.contrast( ROIs_1, ROIs_2, dynamic_ranges_dB );

        end % function objects = PSNR( ROIs_1, ROIs_2, dynamic_ranges_dB )

	end % methods

	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	%% methods (protected and hidden)
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	methods (Access = protected, Hidden)

        %------------------------------------------------------------------
        % evaluate samples (scalar)
        %------------------------------------------------------------------
        function result = evaluate_samples( PSNR, samples_1, samples_2 )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % calling function ensures class processing.metrics.PSNR (scalar) for PSNR

            %--------------------------------------------------------------
            % 2.) compute peak signal-to-noise ratio (PSNR)
            %--------------------------------------------------------------
            result = ( max( samples_1 ) - mean( samples_2 ) ) / PSNR.dynamic_range_dB;

        end % function result = evaluate_samples( PSNR, samples_1, samples_2 )

	end % methods (Access = protected, Hidden)

end % classdef PSNR < processing.metrics.contrast
