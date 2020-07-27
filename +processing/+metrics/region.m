%
% abstract superclass for all regions
%
% author: Martin F. Schiffner
% date: 2020-01-08
% modified: 2020-07-06
%
classdef (Abstract) region < processing.metrics.metric

	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	%% properties
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	properties (SetAccess = private)

        % independent properties
        ROI ( 1, 1 ) scattering.sequences.setups.geometry.shape { mustBeNonempty } = scattering.sequences.setups.geometry.orthotope	% ROI to be inspected
        boundary_dB ( 1, 1 ) double { mustBeNegative, mustBeNonempty } = -6     % boundary value in dB

	end % properties

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	%% methods
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    methods

        %------------------------------------------------------------------
        % constructor
        %------------------------------------------------------------------
        function objects = region( ROIs, boundaries_dB )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % ensure one or two arguments
            narginchk( 1, 2 );

            % property validation functions ensure class scattering.sequences.setups.geometry.shape for ROIs

            % ensure nonempty boundaries_dB
            if nargin < 2 || isempty( boundaries_dB )
                boundaries_dB = -6;
            end

            % property validation functions ensure nonempty negative double for boundaries_dB

            % ensure equal number of dimensions and sizes
            [ ROIs, boundaries_dB ] = auxiliary.ensureEqualSize( ROIs, boundaries_dB );

            %--------------------------------------------------------------
            % 2.) create regions
            %--------------------------------------------------------------
            % constructor of superclass
            objects@processing.metrics.metric( size( ROIs ) );

            % iterate regions
            for index_object = 1:numel( objects )

                % set independent properties
                objects( index_object ).ROI = ROIs( index_object );
                objects( index_object ).boundary_dB = boundaries_dB( index_object );

            end % for index_object = 1:numel( objects )

        end % function objects = region( ROIs, boundaries_dB )

	end % methods

	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	%% methods (protected and hidden)
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	methods (Access = protected, Hidden)

        %------------------------------------------------------------------
        % evaluate metric (scalar)
        %------------------------------------------------------------------
        function results = evaluate_scalar( region, image )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % calling function ensures class processing.metrics.metric (scalar) for region
            % calling function ensures class processing.image (scalar) for image

            % ensure class math.grid_regular
            if ~isa( image.grid, 'math.grid_regular' )
                errorStruct.message = 'image.grid must be math.grid_regular!';
                errorStruct.identifier = 'evaluate_scalar:NoRegularGrid';
                error( errorStruct );
            end

            % ensure valid number of dimensions
            if region.ROI.N_dimensions ~= image.grid.N_dimensions
                errorStruct.message = 'Number of dimensions of region.ROI must equal the number of dimensions of the image grid!';
                errorStruct.identifier = 'evaluate_scalar:DimensionMismatch';
                error( errorStruct );
            end

            %--------------------------------------------------------------
            % 2.) compute region metric (scalar)
            %--------------------------------------------------------------
            % specify cell array for results
            results = cell( 1, image.N_images );

            % detect valid grid points in ROIs
            indicator_roi = iselement( region.ROI, image.grid.positions );

            % iterate images
            for index_image = 1:image.N_images

                %----------------------------------------------------------
                % a) subsampling, logarithmic compression, and thresholding
                %----------------------------------------------------------
                % subsampling
                samples_act = image.samples( indicator_roi, index_image );

                % logarithmic compression and hard thresholding
                samples_act_dB = illustration.dB( samples_act, 20 );
                indicator = ( samples_act_dB >= region.boundary_dB );

                %----------------------------------------------------------
                % b) compute region metric (samples)
                %----------------------------------------------------------
                results{ index_image } = evaluate_samples( region, image.grid.cell_ref.volume, indicator );

            end % for index_image = 1:image.N_images

            % concatenate horizontally
            results = cat( 2, results{ : } );

        end % function results = evaluate_scalar( region, image )

	end % methods (Access = protected, Hidden)

	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	%% methods (abstract, protected, and hidden)
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	methods (Abstract, Access = protected, Hidden)

        %------------------------------------------------------------------
        % evaluate metric (samples)
        %------------------------------------------------------------------
        result = evaluate_samples( region, delta_V, indicator )

	end % methods (Abstract, Access = protected, Hidden)

end % classdef (Abstract) region < processing.metrics.metric
