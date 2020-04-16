%
% abstract superclass for all contrast metrics
%
% author: Martin F. Schiffner
% date: 2020-02-29
% modified: 2020-03-14
%
classdef (Abstract) contrast < processing.metrics.metric

	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	%% properties
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	properties (SetAccess = private)

        % independent properties
        ROI_1 ( 1, 1 ) scattering.sequences.setups.geometry.shape { mustBeNonempty } = scattering.sequences.setups.geometry.orthotope	% reference region of interest
        ROI_2 ( 1, 1 ) scattering.sequences.setups.geometry.shape { mustBeNonempty } = scattering.sequences.setups.geometry.orthotope	% noisy region of interest
        dynamic_range_dB ( 1, 1 ) double { mustBePositive, mustBeNonempty } = 70	% limit for dynamic range (dB)

	end % properties

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	%% methods
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    methods

        %------------------------------------------------------------------
        % constructor
        %------------------------------------------------------------------
        function objects = contrast( ROIs_1, ROIs_2, dynamic_ranges_dB )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % property validation functions ensure class scattering.sequences.setups.geometry.shape for ROIs_1 and ROIs_2

            % ensure nonempty dynamic_ranges_dB
            if nargin < 3 || isempty( dynamic_ranges_dB )
                dynamic_ranges_dB = 70;
            end

            % property validation functions ensure nonempty positive doubles for dynamic_ranges_dB

            % ensure equal number of dimensions and sizes
            [ ROIs_1, ROIs_2, dynamic_ranges_dB ] = auxiliary.ensureEqualSize( ROIs_1, ROIs_2, dynamic_ranges_dB );

            %--------------------------------------------------------------
            % 2.) create contrast metrics
            %--------------------------------------------------------------
            % constructor of superclass
            objects@processing.metrics.metric( size( ROIs_1 ) );

            % iterate contrast metrics
            for index_object = 1:numel( objects )

                % set independent properties
                objects( index_object ).ROI_1 = ROIs_1( index_object );
                objects( index_object ).ROI_2 = ROIs_2( index_object );
                objects( index_object ).dynamic_range_dB = dynamic_ranges_dB( index_object );

            end % for index_object = 1:numel( objects )

        end % function objects = contrast( ROIs_1, ROIs_2, dynamic_ranges_dB )

	end % methods

	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	%% methods (protected and hidden)
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	methods (Access = protected, Hidden)

        %------------------------------------------------------------------
        % evaluate metric (scalar)
        %------------------------------------------------------------------
        function results = evaluate_scalar( contrast, image )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % calling function ensures class processing.metrics.metric (scalar) for contrast
            % calling function ensures class processing.image (scalar) for image

            %--------------------------------------------------------------
            % 2.) compute contrast metric
            %--------------------------------------------------------------
            % specify cell arrays
            results = zeros( 1, image.N_images );

            % detect valid grid points in ROIs
            indicator_roi_ref = iselement( contrast.ROI_1, image.grid.positions );
            indicator_roi_noise = iselement( contrast.ROI_2, image.grid.positions );

            % iterate images
            for index_image = 1:image.N_images

                %----------------------------------------------------------
                % a) logarithmic compression and subsampling
                %----------------------------------------------------------
                % logarithmic compression
                samples_act_dB = illustration.dB( image.samples( :, index_image ), 20 );

                % limit dynamic range
                indicator_dynamic_range = samples_act_dB < - contrast.dynamic_range_dB;
                samples_act_dB( indicator_dynamic_range ) = - contrast.dynamic_range_dB;

                % subsampling
                samples_act_dB_ref = samples_act_dB( indicator_roi_ref );
                samples_act_dB_noise = samples_act_dB( indicator_roi_noise );

                %----------------------------------------------------------
                % b) compute contrast metric (samples)
                %----------------------------------------------------------
                results( index_image ) = evaluate_samples( contrast, samples_act_dB_ref, samples_act_dB_noise );

            end % for index_image = 1:image.N_images

        end % function results = evaluate_scalar( contrast, image )

	end % methods (Access = protected, Hidden)

	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	%% methods (abstract, protected, and hidden)
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	methods (Abstract, Access = protected, Hidden)

        %------------------------------------------------------------------
        % evaluate metric (scalar)
        %------------------------------------------------------------------
        result = evaluate_samples( contrast, samples_act_dB_ref, samples_act_dB_noise )

	end % methods (Abstract, Access = protected, Hidden)

end % classdef (Abstract) contrast < processing.metrics.metric
