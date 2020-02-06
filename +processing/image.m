%
% superclass for all images
%
%
% author: Martin F. Schiffner
% date: 2019-09-10
% modified: 2020-02-04
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

            % single grids / multiple samples
            if isscalar( grids ) && ~isscalar( samples )
                grids = repmat( grids, size( samples ) );
            end

            % ensure equal number of dimensions and sizes
            auxiliary.mustBeEqualSize( grids, samples );

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
                errorStruct.identifier = 'profile:NoImages';
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
                    errorStruct.message = sprintf( 'No implementation for options{ %d }!', index_matrix );
                    errorStruct.identifier = 'evaluate_metric:UnknownProcessingOptions';
                    error( errorStruct );

                end % if isa( options{ index_matrix }, 'processing.options.profile' )

            end % for index_matrix = 1:numel( images )

            % avoid cell array for single images
            if isscalar( images )
                results = results{ 1 };
            end

        end % function results = evaluate_metric( images, options )

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
            % calling function ensures class processing.image for image
            % calling function ensures scalar for image

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
            % extract axes and number of points
            axes = get_axes( image.grid );

            % specify cell array for profiles
            profiles = cell( size( options ) );

            % iterate options
            for index_options = 1:numel( options )

                %----------------------------------------------------------
                % a) project image pixels
                %----------------------------------------------------------
                % ensure valid dimension
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

                    % project samples
                    for index_dim = 2:image.grid.N_dimensions
                        samples_act = vecnorm( samples_act, 2, index_dim );
                    end

                    % save profile
% TODO: physical unit?!
                    profiles{ index_options }{ index_image } = samples_act * sqrt( image.grid.cell_ref.edge_lengths( options( index_options ).dim ) );

                end % for index_image = 1:image.N_images

                % concatenate horizontally
                profiles{ index_options } = cat( 2, profiles{ index_options }{ : } );

                % add zeros
%                 profiles{ index_options } = [ profiles{ index_options }; zeros( options( index_options ).N_zeros_add, image.N_images ) ];

                %----------------------------------------------------------
                % b) create signal matrices
                %----------------------------------------------------------
                if image.N_images == 1
                    profiles{ index_options } = processing.signal( axes_cut( options( index_options ).dim ), profiles{ index_options } );
                else
                    profiles{ index_options } = processing.signal_matrix( axes_cut( options( index_options ).dim ), profiles{ index_options } );
                end

                %----------------------------------------------------------
                % c) interpolation
                %----------------------------------------------------------
                % apply window functions to smooth boundaries before DFT-based interpolation (use original time window)
                profiles{ index_options } = cut_out( profiles{ index_options }, profiles{ index_options }.axis.members( 1 ), profiles{ index_options }.axis.members( end ), [], options( index_options ).setting_window );

                % interpolate profiles
                profiles{ index_options } = interpolate( profiles{ index_options }, options( index_options ).factor_interp );

            end % for index_options = 1:numel( options )

            % create signal matrix array
            profiles = reshape( cat( 1, profiles{ : } ), size( options ) );

        end % function profiles = profile( image, options )

        %------------------------------------------------------------------
        % region extents
        %------------------------------------------------------------------
        function RBs = region_boundary( image, options )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % calling function ensures class processing.image for image
            % calling function ensures scalar for image

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
            axes = get_axes( images( index_matrix ).grid );

            % specify cell arrays
            N_samples = cell( size( options ) );
            volumes = cell( size( options ) );

            % iterate options
            for index_options = 1:numel( options )

                % ensure valid numbers of dimensions
                if options( index_options ).ROI.N_dimensions ~= images( index_matrix ).grid.N_dimensions
                    errorStruct.message = sprintf( 'Numbers of dimensions in options{ %d }( %d ).ROI and images( %d ).grid must equal!', index_matrix, index_options, index_matrix );
                    errorStruct.identifier = 'region_boundary:DimensionMismatch';
                    error( errorStruct );
                end

                % cut out axes
                [ ~, indicators ] = cut_out( axes, cat( 2, options( index_options ).ROI.intervals.lb ), cat( 2, options( index_options ).ROI.intervals.ub ) );

                % initialize results w/ zeros
                N_samples{ index_options } = zeros( 1, images( index_matrix ).N_images );
                volumes{ index_options } = zeros( 1, images( index_matrix ).N_images );

                % iterate images
                for index_image = 1:images( index_matrix ).N_images

                    % extract relevant samples
                    samples_act = reshape( images( index_matrix ).samples( :, index_image ), images( index_matrix ).grid.N_points_axis );
                    samples_act = samples_act( indicators{ : } );

                    % logarithmic compression and hard thresholding
                    samples_act_dB = illustration.dB( samples_act, 20 );
                    indicator = ( samples_act_dB >= options( index_options ).boundary_dB );

                    % number of samples and volumes
                    N_samples{ index_options }( index_image ) = sum( indicator( : ) );
                    volumes{ index_options }( index_image ) = N_samples{ index_options }( index_image ) * images( index_matrix ).grid.cell_ref.volume;

                end % for index_image = 1:images( index_matrix ).N_images

            end % for index_options = 1:numel( options )

            % create structures
            RBs = struct( 'N_samples', N_samples, 'volume', volumes );

        end % function RBs = region_boundary( image, options )

        %------------------------------------------------------------------
        % contrast-to-noise ratios (CNRs)
        %------------------------------------------------------------------
        function CNRs = contrast_noise_ratios( image, options )
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
            % REFERENCES:
            %	[1] J. T. Bushberg, J. A. Seibert, E. M. Leidholdt, and J. M. Boone, "The Essential Physics of Medical Imaging", Sect. 4.8

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % calling function ensures class processing.image for image
            % calling function ensures scalar for image

            % calling function ensures class processing.options.contrast for options

            %--------------------------------------------------------------
            % 2.) compute contrast-to-noise ratios (CNRs)
            %--------------------------------------------------------------
            % specify cell arrays
            CNRs = cell( size( options ) );

            % iterate options
            for index_options = 1:numel( options )

                % detect valid grid points in ROIs
                [ ~, indicator_roi_ref ] = cut_out( image.grid, options( index_options ).ROI_ref );
                [ ~, indicator_roi_noise ] = cut_out( image.grid, options( index_options ).ROI_noise );

                % initialize CNRs w/ zeros
                CNRs{ index_options } = zeros( image.N_images, 1 );

                % iterate images
                for index_image = 1:image.N_images

                    % logarithmic compression
                    samples_act_dB = illustration.dB( image.samples( :, index_image ), 20 );

                    % limit dynamic range
                    indicator_dynamic_range = samples_act_dB < - options( index_options ).dynamic_range_dB;
                    samples_act_dB( indicator_dynamic_range ) = - options( index_options ).dynamic_range_dB;

                    % subsampling
                    samples_act_dB_ref = samples_act_dB( indicator_roi_ref );
                    samples_act_dB_noise = samples_act_dB( indicator_roi_noise );

                    % statistics
                    samples_act_dB_mean = mean( samples_act_dB_ref );
                    samples_act_dB_var = var( samples_act_dB_ref );
                    samples_act_dB_noise_mean = mean( samples_act_dB_noise );
                    samples_act_dB_noise_var = var( samples_act_dB_noise );

                    % contrast-to-noise ratio (CNR) % TODO: 20 * log10( ) ?
                    CNRs{ index_options }( index_image ) = abs( samples_act_dB_mean - samples_act_dB_noise_mean ) / sqrt( ( samples_act_dB_var + samples_act_dB_noise_var ) / 2 );

                end % for index_image = 1:image.N_images

            end % for index_options = 1:numel( options )

            % avoid cell array for single options
            if isscalar( options )
                CNRs = CNRs{ 1 };
            end

        end % function CNRs = contrast_noise_ratios( image, options )

        %------------------------------------------------------------------
        % speckle quality (from Kolmogorov-Smirnov test)
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
