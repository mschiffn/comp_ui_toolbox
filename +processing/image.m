%
% superclass for all images
%
%
% author: Martin F. Schiffner
% date: 2019-09-10
% modified: 2020-06-29
%
% TODO: make subclass of field
%
classdef image

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% properties
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	properties (SetAccess = protected)

        % independent properties
        grid ( 1, 1 ) math.grid
        samples ( :, : ) double

        % dependent properties
        N_images ( 1, 1 ) double { mustBeInteger, mustBePositive } = 1

    end % properties

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% methods
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	methods

        %------------------------------------------------------------------
        % constructor
        %------------------------------------------------------------------
        function objects = image( grids, samples )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % ensure two arguments
            narginchk( 2, 2 );

            % ensure class math.grid
            if ~isa( grids, 'math.grid' )
                errorStruct.message = 'grids must be math.grid!';
                errorStruct.identifier = 'image:NoGrids';
                error( errorStruct );
            end

            % ensure cell array for samples
            if ~iscell( samples )
                samples = { samples };
            end

            % ensure equal number of dimensions and sizes
            [ grids, samples ] = auxiliary.ensureEqualSize( grids, samples );

            %--------------------------------------------------------------
            % 2.) create images
            %--------------------------------------------------------------
            % repeat default image
            objects = repmat( objects, size( grids ) );

            % iterate images
            for index_object = 1:numel( objects )

                % ensure numeric matrix
                if ~( isnumeric( samples{ index_object } ) && ismatrix( samples{ index_object } ) )
                    errorStruct.message = sprintf( 'samples{ %d } must be a numeric matrix!', index_object );
                    errorStruct.identifier = 'image:NoNumericMatrix';
                    error( errorStruct );
                end

                % ensure correct sizes
                if grids( index_object ).N_points ~= size( samples{ index_object }, 1 )
                    errorStruct.message = sprintf( 'Size of samples{ %d } along the first dimension must match grids( %d ).N_points!', index_object, index_object );
                    errorStruct.identifier = 'image:SizeMismatch';
                    error( errorStruct );
                end

                % set independent properties
                objects( index_object ).grid = grids( index_object );
                objects( index_object ).samples = samples{ index_object };

                % set dependent properties
                objects( index_object ).N_images = size( objects( index_object ).samples, 2 );

            end % for index_object = 1:numel( objects )

        end % function objects = image( grids, samples )

        %------------------------------------------------------------------
        % evaluate specified metric (wrapper)
        %------------------------------------------------------------------
        function results = evaluate_metric( images, options )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % ensure class processing.image
            if ~isa( images, 'processing.image' )
                errorStruct.message = 'images must be processing.image!';
                errorStruct.identifier = 'evaluate_metric:NoImages';
                error( errorStruct );
            end

            % ensure cell array for options
            if ~iscell( options )
                options = { options };
            end

            % multiple images / single options
            if ~isscalar( images ) && isscalar( options )
                options = repmat( options, size( images ) );
            end

            % single images / multiple options
            if isscalar( images ) && ~isscalar( options )
                images = repmat( images, size( options ) );
            end

            % ensure equal number of dimensions and sizes
            auxiliary.mustBeEqualSize( images, options );

            %--------------------------------------------------------------
            % 2.) compute metrics
            %--------------------------------------------------------------
            % specify cell array for results
            results = cell( size( images ) );

            % iterate image matrices
            for index_matrix = 1:numel( images )

                %----------------------------------------------------------
                % a) check arguments
                %----------------------------------------------------------
                % ensure class processing.options.template
%                 if ~isa( options{ index_matrix }, 'processing.options.template' )
%                     errorStruct.message = sprintf( 'options{ %d } must be processing.options.template!', index_matrix );
%                     errorStruct.identifier = 'profile:NoProcessingOptions';
%                     error( errorStruct );
%                 end

                %----------------------------------------------------------
                % b) compute metrics
                %----------------------------------------------------------
                if isa( options{ index_matrix }, 'processing.options.profile' )

                    %------------------------------------------------------
                    % i.) projected profiles
                    %------------------------------------------------------
                    results{ index_matrix } = profile( images( index_matrix ), options{ index_matrix } );

                elseif isa( options{ index_matrix }, 'processing.options.region' )

                    %------------------------------------------------------
                    % ii.) region extents
                    %------------------------------------------------------
                    results{ index_matrix } = region_boundary( images( index_matrix ), options{ index_matrix } );

                elseif isa( options{ index_matrix }, 'processing.options.contrast' )

                    %------------------------------------------------------
                    % iii.) contrast-to-noise ratios (CNRs)
                    %------------------------------------------------------
                    results{ index_matrix } = contrast_noise_ratios( images( index_matrix ), options{ index_matrix } );

                elseif isa( options{ index_matrix }, 'processing.options.speckle' )

                    %------------------------------------------------------
                    % iv.) speckle quality
                    %------------------------------------------------------
                    results{ index_matrix } = speckle_quality( images( index_matrix ), options{ index_matrix } );

                else

                    %------------------------------------------------------
                    % v.) unknown options
                    %------------------------------------------------------
                    errorStruct.message = sprintf( 'No implementation for options{ %d } (%s)!', index_matrix, class( options{ index_matrix } ) );
                    errorStruct.identifier = 'evaluate_metric:UnknownProcessingOptions';
                    error( errorStruct );

                end % if isa( options{ index_matrix }, 'processing.options.profile' )

            end % for index_matrix = 1:numel( images )

            % avoid cell array for single images
            if isscalar( images )
                results = results{ 1 };
            end

        end % function results = evaluate_metric( images, options )

        %------------------------------------------------------------------
        % show
        %------------------------------------------------------------------
        function show( images, dynamic_ranges_dB )

            samples_act = reshape( images.samples, images.grid.N_points_axis );
            imagesc( illustration.dB( samples_act, 20 ), [ -dynamic_ranges_dB, 0 ] );

        end

	end % methods

	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	%% methods (private and hidden)
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	methods (Access = private, Hidden)

        %------------------------------------------------------------------
        % projected profile
        %------------------------------------------------------------------
        function profiles = profile( image, options )

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

            %--------------------------------------------------------------
            % 2.) compute projected profiles
            %--------------------------------------------------------------
            % extract axes
            axes = get_axes( image.grid );

            % specify cell array for profiles
            profiles = cell( size( options ) );

            % iterate options
            for index_options = 1:numel( options )

                %----------------------------------------------------------
                % a) project image pixels
                %----------------------------------------------------------
                % ensure valid number of dimensions
                if options( index_options ).ROI.N_dimensions ~= image.grid.N_dimensions
                    errorStruct.message = sprintf( 'Number of dimensions of options( %d ).ROI must equal the number of dimensions of the image grid!', index_options );
                    errorStruct.identifier = 'profile:DimensionMismatch';
                    error( errorStruct );
                end

                % cut out axes
                [ axes_cut, indicators ] = cut_out( axes, cat( 2, options( index_options ).ROI.intervals.lb ), cat( 2, options( index_options ).ROI.intervals.ub ) );

                % specify cell array for profiles
                profiles{ index_options } = cell( 1, image.N_images );

                % iterate images
                for index_image = 1:image.N_images

                    % extract relevant samples
                    samples_act = reshape( image.samples( :, index_image ), image.grid.N_points_axis );
                    samples_act = shiftdim( samples_act( indicators{ : } ), options( index_options ).dim - 1 );
                    deltas = circshift( [ axes.delta ], 1 - options( index_options ).dim );

                    % project samples
                    for index_dim = 2:image.grid.N_dimensions
                        samples_act = vecnorm( samples_act, 2, index_dim );
                    end

                    % save profile
                    profiles{ index_options }{ index_image } = samples_act * sqrt( prod( deltas( 2:end ) ) );

                end % for index_image = 1:image.N_images

                %----------------------------------------------------------
                % b) create profile signals / signal matrices
                %----------------------------------------------------------
                % create profile signals
                profiles{ index_options } = processing.signal( axes_cut( options( index_options ).dim ), profiles{ index_options } );

                % try to merge profile signals
                try
                    profiles{ index_options } = merge( profiles{ index_options } );
                catch
                end

                %----------------------------------------------------------
                % c) interpolation
                %----------------------------------------------------------
                % apply window functions to smooth boundaries before DFT-based interpolation (use original time window)
                profiles{ index_options } = cut_out( profiles{ index_options }, profiles{ index_options }.axis.members( 1 ), profiles{ index_options }.axis.members( end ), [], options( index_options ).setting_window );

                % interpolate profiles
                profiles{ index_options } = interpolate( profiles{ index_options }, options( index_options ).factor_interp );

            end % for index_options = 1:numel( options )

            % create signal_matrix or signal array
            profiles = reshape( cat( 1, profiles{ : } ), size( options ) );

        end % function profiles = profile( image, options )

        %------------------------------------------------------------------
        % region extents
        %------------------------------------------------------------------
        function RBs = region_boundary( image, options )

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

            % calling function ensures class processing.options.region for options

            %--------------------------------------------------------------
            % 2.) compute regions
            %--------------------------------------------------------------
            % extract axes
            axes = get_axes( image.grid );

            % specify cell arrays
            N_samples = cell( numel( options ), 1 );
            volumes = cell( numel( options ), 1 );

            % iterate options
            for index_options = 1:numel( options )

                % ensure valid number of dimensions
                if options( index_options ).ROI.N_dimensions ~= image.grid.N_dimensions
                    errorStruct.message = sprintf( 'Number of dimensions of options( %d ).ROI must equal the number of dimensions of the image grid!', index_options );
                    errorStruct.identifier = 'profile:DimensionMismatch';
                    error( errorStruct );
                end

                % cut out axes
                [ ~, indicators ] = cut_out( axes, cat( 2, options( index_options ).ROI.intervals.lb ), cat( 2, options( index_options ).ROI.intervals.ub ) );

                % initialize results w/ zeros
                N_samples{ index_options } = zeros( 1, image.N_images );
                volumes{ index_options } = repmat( image.grid.cell_ref.volume, [ 1, image.N_images ] );

                % iterate images
                for index_image = 1:image.N_images

                    % extract relevant samples
                    samples_act = reshape( image.samples( :, index_image ), image.grid.N_points_axis );
                    samples_act = samples_act( indicators{ : } );

                    % logarithmic compression and hard thresholding
                    samples_act_dB = illustration.dB( samples_act, 20 );
                    indicator = ( samples_act_dB >= options( index_options ).boundary_dB );

                    % number of samples and volumes
                    N_samples{ index_options }( index_image ) = sum( indicator( : ) );
                    volumes{ index_options }( index_image ) = N_samples{ index_options }( index_image ) * image.grid.cell_ref.volume;

                end % for index_image = 1:image.N_images

            end % for index_options = 1:numel( options )

            % create structures
            RBs = struct( 'N_samples', N_samples, 'volume', volumes );

        end % function RBs = region_boundary( image, options )

        %------------------------------------------------------------------
        % contrast
        %------------------------------------------------------------------
        function results = contrast_noise_ratios( image, options )
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

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % calling function ensures class processing.image (scalar) for image

            % calling function ensures class processing.options.contrast for options

            %--------------------------------------------------------------
            % 2.) compute contrast-to-noise ratios (CNRs)
            %--------------------------------------------------------------
            % specify cell arrays
            results = repmat( struct,  [ numel( options ), image.N_images ] );

            % iterate options
            for index_options = 1:numel( options )

                % detect valid grid points in ROIs
                [ ~, indicator_roi_ref ] = cut_out( image.grid, options( index_options ).ROI_ref );
                [ ~, indicator_roi_noise ] = cut_out( image.grid, options( index_options ).ROI_noise );

                % iterate images
                for index_image = 1:image.N_images

                    %------------------------------------------------------
                    % a) logarithmic compression and subsampling
                    %------------------------------------------------------
                    % logarithmic compression
                    samples_act_dB = illustration.dB( image.samples( :, index_image ), 20 );

                    % limit dynamic range
                    indicator_dynamic_range = samples_act_dB < - options( index_options ).dynamic_range_dB;
                    samples_act_dB( indicator_dynamic_range ) = - options( index_options ).dynamic_range_dB;

                    % subsampling
                    samples_act_dB_ref = samples_act_dB( indicator_roi_ref );
                    samples_act_dB_noise = samples_act_dB( indicator_roi_noise );

                    %------------------------------------------------------
                    % b) generalized contrast-to-noise ratio (gCNR)
                    %------------------------------------------------------
                    % estimate PDFs of samples_act_dB
                    samples_act_dB_ref_pdf = histcounts( samples_act_dB_ref, (-options( index_options ).dynamic_range_dB:0), 'Normalization', 'pdf' );
                    samples_act_dB_noise_pdf = histcounts( samples_act_dB_noise, (-options( index_options ).dynamic_range_dB:0), 'Normalization', 'pdf' );

                    % overlap of PDFs and gCNR
                    overlap = sum( min( samples_act_dB_ref_pdf, samples_act_dB_noise_pdf ) );
                    results( index_options, index_image ).gCNR = 1 - overlap;

                    %------------------------------------------------------
                    % c) contrast-to-noise ratio (CNR)
                    %------------------------------------------------------
                    % means and variances
                    samples_act_dB_ref_mean = mean( samples_act_dB_ref );
                    samples_act_dB_ref_var = var( samples_act_dB_ref );
                    samples_act_dB_noise_mean = mean( samples_act_dB_noise );
                    samples_act_dB_noise_var = var( samples_act_dB_noise );

                    % contrast-to-noise ratio (CNR)
                    results( index_options, index_image ).CNR = abs( samples_act_dB_ref_mean - samples_act_dB_noise_mean ) / sqrt( ( samples_act_dB_ref_var + samples_act_dB_noise_var ) / 2 );

                end % for index_image = 1:image.N_images

            end % for index_options = 1:numel( options )

        end % function results = contrast_noise_ratios( image, options )

        %------------------------------------------------------------------
        % speckle quality (Kolmogorov-Smirnov test)
        %------------------------------------------------------------------
        function CNRs = speckle_quality( image, options )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % calling function ensures class processing.image for image
            % calling function ensures scalar for image

            % calling function ensures class processing.options.speckle for options

            %--------------------------------------------------------------
            % 2.) evaluate speckle quality
            %--------------------------------------------------------------
            % specify cell arrays
            CNRs = cell( size( options ) );

            % iterate options
            for index_options = 1:numel( options )

                % detect valid grid points in ROIs
                [ ~, indicator_roi_ref ] = cut_out( image.grid, options( index_options ).ROI_ref );

                % iterate images
                for index_image = 1:image.N_images
                end

                                        %-- Compute mask inside
                    x = h.pht.RoiCenterX(k);
                    z = h.pht.RoiCenterZ(k);
                    

                    %-- Extract corresponding enveloppe ROI
                    [idzz,idxx] = find(maskROI==k);
                    envRoi = env(min(idzz):max(idzz),min(idxx):max(idxx));
                    
                    %-- Downsample the block to ensure statistical independency
                    sample = envRoi(1:5:end,1:5:end);
                    sample = sample(:);         
                    
                    %-- Empirical evaluation of the variance on the selected block
                    var_block = tools.get_rayleigh_param(sample);
                    
                    %-- Application of the KS test against the Rayleigh pdf with 5% confidence interval
                    testres = 1-kstest(sample, 'CDF', [sample, raylcdf(sample, sqrt(var_block))], 'alpha', 0.05);
                    
                    %-- Ploting Roi contour along with a code color
                    %-- green: test passed || red: test failed                    
                    if (h.flagDisplay==1)
                        figure(1);
                        if (testres==0)
                            hold on; contour(h.scan.x_axis*1e3,h.scan.z_axis*1e3,maskROI,[1 1],'r-','linewidth',2);
                        else
                            hold on; contour(h.scan.x_axis*1e3,h.scan.z_axis*1e3,maskROI,[1 1],'g-','linewidth',2);
                        end
                        pause(0.1);
                    end
                    h.score(f,k) = testres;
                    %-- Function which estimates the parameter of the Rayleigh distribution
    %-- Function prototype: sig = get_rayleigh_param(sample)
    
    %-- Authors: Olivier Bernard (olivier.bernard@creatis.insa-lyon.fr)
    %--          Alfonso Rodriguez-Molares (alfonso.r.molares@ntnu.no)

    %-- $Date: 2016/03/01 $  

    sample = sample(:);
    M = sum( power(sample,2) );
    N = length(sample);
    sig = M / (2*N);

            end % for index_options = 1:numel( options )

            % avoid cell array for single options
            if isscalar( options )
                CNRs = CNRs{ 1 };
            end

        end % function CNRs = speckle_quality( image, options )

	end % methods (Access = private, Hidden)

end % classdef image
