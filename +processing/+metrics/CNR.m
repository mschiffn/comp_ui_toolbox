%
% superclass for all contrast-to-noise ratios (CNRs)
%
% author: Martin F. Schiffner
% date: 2020-02-29
% modified: 2020-03-14
%
classdef CNR < processing.metrics.contrast

	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	%% methods
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	methods

        %------------------------------------------------------------------
        % constructor
        %------------------------------------------------------------------
        function objects = CNR( ROIs_1, ROIs_2, dynamic_ranges_dB )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % superclass validation functions ensure class math.orthotope for ROIs_1 and ROIs_2

            % ensure equal subclasses of physical_values.length
%             auxiliary.mustBeEqualSubclasses( 'physical_values.length', ROIs_1.intervals.lb );

            %--------------------------------------------------------------
            % 2.) create contrast-to-noise ratios (CNRs)
            %--------------------------------------------------------------
            % constructor of superclass
            objects@processing.metrics.contrast( ROIs_1, ROIs_2, dynamic_ranges_dB );

        end % function objects = CNR( ROIs_1, ROIs_2, dynamic_ranges_dB )

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
            % calling function ensures class processing.metrics.CNR (scalar) for CNR

            %--------------------------------------------------------------
            % 2.) compute contrast-to-noise ratio (CNR)
            %--------------------------------------------------------------
            % means and variances
            samples_1_mean = mean( samples_1 );
            samples_1_var = var( samples_1 );
            samples_2_mean = mean( samples_2 );
            samples_2_var = var( samples_2 );

            % contrast-to-noise ratio (CNR)
            result = abs( samples_1_mean - samples_2_mean ) / sqrt( ( samples_1_var + samples_2_var ) / 2 );

        end % function result = evaluate_samples( ~, samples_1, samples_2 )

	end % methods (Access = protected, Hidden)

end % classdef CNR < processing.metrics.contrast
