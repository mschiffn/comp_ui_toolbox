%
% abstract superclass for all regions
%
% author: Martin F. Schiffner
% date: 2020-01-08
% modified: 2020-03-13
%
classdef (Abstract) region < processing.metrics.metric

	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	%% properties
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	properties (SetAccess = private)

        % independent properties
        ROI ( 1, 1 ) math.orthotope { mustBeNonempty } = math.orthotope         % ROI to be inspected
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
            % ensure class math.orthotope
            if ~isa( ROIs, 'math.orthotope' )
                errorStruct.message = 'ROIs must be math.orthotope!';
                errorStruct.identifier = 'region:NoOrthotopes';
                error( errorStruct );
            end

            % property validation functions ensure valid boundaries_dB

            % multiple ROIs / single boundaries_dB
            if ~isscalar( ROIs ) && isscalar( boundaries_dB )
                boundaries_dB = repmat( boundaries_dB, size( ROIs ) );
            end

            % single ROIs / multiple boundaries_dB
            if isscalar( ROIs ) && ~isscalar( boundaries_dB )
                ROIs = repmat( ROIs, size( boundaries_dB ) );
            end

            % ensure equal number of dimensions and sizes
            auxiliary.mustBeEqualSize( ROIs, boundaries_dB );

            %--------------------------------------------------------------
            % 2.) create regions
            %--------------------------------------------------------------
            % repeat default region
            objects@processing.metrics.metric( size( boundaries_dB ) );

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

            % ensure class math.grid_regular_orthogonal
            if ~isa( image.grid, 'math.grid_regular_orthogonal' )
                errorStruct.message = 'image.grid must be math.grid_regular_orthogonal!';
                errorStruct.identifier = 'evaluate_scalar:NoOrthogonalRegularGrid';
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
            % extract axes
            axes = get_axes( image.grid );

            % cut out axes
            [ ~, indicators ] = cut_out( axes, cat( 2, region.ROI.intervals.lb ), cat( 2, region.ROI.intervals.ub ) );

            % specify cell array for results
            results = cell( 1, image.N_images );

            % iterate images
            for index_image = 1:image.N_images

                %----------------------------------------------------------
                % a) subsampling, logarithmic compression, and thresholding
                %----------------------------------------------------------
                % extract relevant samples
                samples_act = reshape( image.samples( :, index_image ), image.grid.N_points_axis );
                samples_act = samples_act( indicators{ : } );

                % logarithmic compression and hard thresholding
                samples_act_dB = illustration.dB( samples_act, 20 );
                indicator = ( samples_act_dB >= options( index_options ).boundary_dB );

                %----------------------------------------------------------
                % b) compute region metric (samples)
                %----------------------------------------------------------
                results{ index_image } = evaluate_samples( region, indicator );

            end % for index_image = 1:image.N_images

            % concatenate horizontally
            results = cat( 2, results{ : } );

        end % function results = evaluate_scalar( contrast, image )

	end % methods (Access = protected, Hidden)

	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	%% methods (abstract, protected, and hidden)
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	methods (Abstract, Access = protected, Hidden)

        %------------------------------------------------------------------
        % evaluate metric (samples)
        %------------------------------------------------------------------
        result = evaluate_samples( contrast, samples_act_dB_ref, samples_act_dB_noise )

	end % methods (Abstract, Access = protected, Hidden)

end % classdef (Abstract) region < processing.metrics.metric
