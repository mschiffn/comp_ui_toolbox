%
% superclass for all projected profiles
%
% author: Martin F. Schiffner
% date: 2020-01-08
% modified: 2020-10-13
%
classdef profile < processing.metrics.metric

	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	%% properties
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	properties (SetAccess = private)

        % independent properties
        ROI ( 1, 1 ) math.orthotope { mustBeNonempty } = math.orthotope                                 % region of interest
        dim ( 1, 1 ) double { mustBePositive, mustBeInteger, mustBeNonempty } = 1                       % profile dimension (projection along remaining dimensions)
        N_zeros_add ( 1, 1 ) double { mustBeNonnegative, mustBeInteger, mustBeNonempty } = 50           % number of padded zeros
        factor_interp ( 1, 1 ) double { mustBePositive, mustBeInteger, mustBeNonempty } = 20            % interpolation factor
        setting_window ( 1, 1 ) auxiliary.setting_window { mustBeNonempty } = auxiliary.setting_window	% window settings

    end % properties

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	%% methods
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    methods

        %------------------------------------------------------------------
        % constructor
        %------------------------------------------------------------------
        function objects = profile( ROIs, dims )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % ensure two arguments
            narginchk( 2, 2 );

            % ensure class math.orthotope
            if ~isa( ROIs, 'math.orthotope' )
                errorStruct.message = 'ROIs must be math.orthotope!';
                errorStruct.identifier = 'profile:NoOrthotopes';
                error( errorStruct );
            end

            % ensure equal number of dimensions and sizes
            [ ROIs, dims ] = auxiliary.ensureEqualSize( ROIs, dims );

            %--------------------------------------------------------------
            % 2.) create profile options
            %--------------------------------------------------------------
            % constructor of superclass
            objects@processing.metrics.metric( size( ROIs ) );

            % iterate profile options
            for index_object = 1:numel( objects )

                % ensure equal subclasses of physical_values.length
                auxiliary.mustBeEqualSubclasses( 'physical_values.length', ROIs( index_object ).intervals.lb );

                % ensure valid profile dimension
                if ~ismember( dims( index_object ), ( 1:ROIs( index_object ).N_dimensions ) )
                    errorStruct.message = sprintf( 'dims( %d ) must be greater than or equal to 1 but smaller than or equalt to %d!', index_object, ROIs( index_object ).N_dimensions );
                    errorStruct.identifier = 'profile:InvalidDims';
                    error( errorStruct );
                end

                % set independent properties
                objects( index_object ).ROI = ROIs( index_object );
                objects( index_object ).dim = dims( index_object );

            end % for index_object = 1:numel( objects )

        end % function objects = profile( ROIs, dims )

	end % methods

	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	%% methods (protected and hidden)
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	methods (Access = protected, Hidden)

        %------------------------------------------------------------------
        % evaluate metric (scalar)
        %------------------------------------------------------------------
        function results = evaluate_scalar( profile, image )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % calling function ensures class processing.image (scalar) for image

            % ensure class math.grid_regular_orthogonal
            if ~isa( image.grid, 'math.grid_regular_orthogonal' )
                errorStruct.message = 'image.grid must be math.grid_regular_orthogonal!';
                errorStruct.identifier = 'profile:NoOrthogonalRegularGrid';
                error( errorStruct );
            end

            % calling function ensures class processing.options.profile for options

            % ensure valid number of dimensions
            if profile.ROI.N_dimensions ~= image.grid.N_dimensions
                errorStruct.message = sprintf( 'Number of dimensions of options( %d ).ROI must equal the number of dimensions of the image grid!', index_options );
                errorStruct.identifier = 'profile:DimensionMismatch';
                error( errorStruct );
            end

            %--------------------------------------------------------------
            % 2.) compute projected profiles
            %--------------------------------------------------------------
            % extract axes
            axes = get_axes( image.grid );

            % cut out axes
            [ axes_cut, indicators ] = cut_out( axes, cat( 2, profile.ROI.intervals.lb ), cat( 2, profile.ROI.intervals.ub ) );

            % specify cell array for results
            results = cell( 1, image.N_images );                

            % iterate images
            for index_image = 1:image.N_images

                %----------------------------------------------------------
                % a) project image pixels
                %----------------------------------------------------------
                % extract relevant samples
                samples_act = reshape( image.samples( :, index_image ), image.grid.N_points_axis );
                samples_act = shiftdim( samples_act( indicators{ : } ), profile.dim - 1 );
                deltas = circshift( [ axes.delta ], 1 - profile.dim );

                % project samples
                for index_dim = 2:image.grid.N_dimensions
                    samples_act = vecnorm( samples_act, 2, index_dim );
                end

                % save profile
                results{ index_image } = samples_act * sqrt( prod( deltas( 2:end ) ) );

            end % for index_image = 1:image.N_images

            %--------------------------------------------------------------
            % b) create profile signals / signal matrices
            %--------------------------------------------------------------
            % create profile signals
            results = processing.signal( axes_cut( profile.dim ), results );

            % try to merge profile signals
            try
                results = merge( results );
            catch
            end

            %--------------------------------------------------------------
            % c) interpolation
            %--------------------------------------------------------------
            % apply window functions to smooth boundaries before DFT-based interpolation (use original time window)
            results = cut_out( results, results.axis.members( 1 ), results.axis.members( end ), [], profile.setting_window );

            % interpolate profiles
            results = interpolate( results, profile.factor_interp );

        end % function results = evaluate_scalar( profile, image )

	end % methods (Access = protected, Hidden)

end % classdef profile < processing.metrics.metric
