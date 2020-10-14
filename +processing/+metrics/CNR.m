%
% superclass for all contrast-to-noise ratios (CNRs)
%
%   "The contrast-to-noise ratio (CNR) is
%    an object size-independent measure of the signal level in the presence of noise.
%    Take the example of a disk as the object (Fig. 4-33).
%    The contrast in this example is the difference between
%    [1.)] the average gray scale of a region of interest (ROI) in the disk ( \bar{x}_{ \text{S} } ) and
%    [2.)] that in an ROI in the background ( \bar{x}_{ \text{BG} } ), and
%    the noise can be calculated from the background ROI as well.
%    Thus, the CNR is given by
%    [ CNR = ( \bar{x}_{ \text{S} } - \bar{x}_{ \text{bg} } ) / \sigma_{ \text{bg} } ] (4-22)" (see [1, p. 91]
%
%   "Intuitively, we assume that higher CNR leads to higher probability of lesion detection, and this is indeed the case for DAS" (see [2])
%
% REFERENCES:
%	[1] J. T. Bushberg, J. A. Seibert, E. M. Leidholdt, and J. M. Boone, "The Essential Physics of Medical Imaging", Sect. 4.8
%	[2] A. Rodriguez-Molares, O. M. H. Rindal, J. D’hooge, S.-E. Måsøy, A. Austeng, and H. Torp, "The Generalized Contrast-to-Noise Ratio",
%       2018 IEEE Int. Ultrasonics Symp. (IUS), 2018, DOI: 10.1109/ULTSYM.2018.8580101
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
