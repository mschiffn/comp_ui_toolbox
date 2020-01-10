%
% superclass for all images
%
% author: Martin F. Schiffner
% date: 2019-09-10
% modified: 2020-01-10
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
        % peaks
        %------------------------------------------------------------------
        function positions_peaks = peaks( images, s )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % ensure class processing.image
            if ~isa( images, 'processing.image' )
                errorStruct.message = 'images must be processing.image!';
                errorStruct.identifier = 'peaks:NoImages';
                error( errorStruct );
            end

            % multiple images / single s
            if ~isscalar( images ) && isscalar( s )
                s = repmat( s, size( images ) );
            end

            % ensure equal number of dimensions and sizes
            auxiliary.mustBeEqualSize( images, s );

            %--------------------------------------------------------------
            % 2.) determine peaks
            %--------------------------------------------------------------
            % specify cell array
            positions_peaks = cell( size( images ) );

            % iterate image matrices
            for index_image = 1:numel( images )

                [ samples_sorted, indices ] = sort( abs( images( index_image ).samples ), 'descend' );
                positions_peaks{ index_image } = images( index_image ).grid.positions( indices( 1:s( index_image ) ), : );

            end

            % avoid cell array for single images
            if isscalar( images )
                positions_peaks = positions_peaks{ 1 };
            end

        end % function positions_peaks = peaks( images, s )

        %------------------------------------------------------------------
        % projected profile
        %------------------------------------------------------------------
        function profiles = profile( images, options )

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
            % 2.) compute projected profiles
            %--------------------------------------------------------------
            % specify cell array for profiles
            profiles = cell( size( images ) );

            % iterate image matrices
            for index_image = 1:numel( images )

                %----------------------------------------------------------
                % a) check arguments
                %----------------------------------------------------------
                % ensure class math.grid_regular_orthogonal
                if ~isa( images( index_image ).grid, 'math.grid_regular_orthogonal' )
                    errorStruct.message = sprintf( 'images( %d ).grid must be math.grid_regular_orthogonal!', index_image );
                    errorStruct.identifier = 'profile:NoOrthogonalRegularGrid';
                    error( errorStruct );
                end

                % ensure class processing.options.profile
                if ~isa( options{ index_image }, 'processing.options.profile' )
                    errorStruct.message = sprintf( 'options{ %d } must be processing.options.profile!', index_image );
                    errorStruct.identifier = 'profile:NoOptions';
                    error( errorStruct );
                end

                %----------------------------------------------------------
                % b) compute profiles in specified window
                %----------------------------------------------------------
                % extract axes and number of points
% TODO: maintain regularity of axis!
                axes = get_axes( images( index_image ).grid );
                N_points_axis_act = images( index_image ).grid.N_points_axis;

                % specify cell array for profiles
                profiles{ index_image } = cell( size( options{ index_image } ) );

                % iterate options
                for index_options = 1:numel( options{ index_image } )

                    %------------------------------------------------------
                    % i.) project image pixels
                    %------------------------------------------------------
                    mustBeLessThanOrEqual( options{ index_image }( index_options ).dim, images( index_image ).grid.N_dimensions );

                    % cut out axis
                    [ ~, indicator ] = cut_out( axes( options{ index_image }( index_options ).dim ), options{ index_image }( index_options ).interval.lb, options{ index_image }( index_options ).interval.ub );

                    %
                    str_selector = repmat( { ':' }, [ 1, images( index_image ).grid.N_dimensions ] );
                    str_selector{ options{ index_image }( index_options ).dim } = indicator;

                    % specify cell array for profiles
                    profiles{ index_image }{ index_options } = cell( 1, images( index_image ).N_images );

                    % iterate images
                    for index_col = 1:images( index_image ).N_images

                        samples_act = reshape( images( index_image ).samples( :, index_col ), images( index_image ).grid.N_points_axis );
                        profiles{ index_image }{ index_options }{ index_col } = squeeze( vecnorm( samples_act( str_selector{ : } ), 2, options{ index_image }( index_options ).dim ) ) * sqrt( images( index_image ).grid.cell_ref.edge_lengths( options{ index_image }( index_options ).dim ) );

                    end % for index_col = 1:images( index_image ).N_images

%                     indicator = ( images( index_image ).grid.positions( :, options{ index_image }( index_options ).dim ) >= intervals_pos( index_image ).lb ) & ( images( index_image ).grid.positions( :, options{ index_image }( index_options ).dim ) <= intervals_pos( index_image ).ub );
%                     N_points_axis_act( options{ index_image }( index_options ).dim ) = sum( indicator ) * N_points_axis_act( options{ index_image }( index_options ).dim ) / images( index_image ).grid.N_points;

                    %------------------------------------------------------
                    % ii.) create signal matrices
                    %------------------------------------------------------
                    % TODO: check number of dimensions
                    N_points_axis_act( options{ index_image }( index_options ).dim ) = 1;

                    indicator = N_points_axis_act > 1;
                    if images( index_image ).N_images == 1
                        profiles{ index_image }{ index_options } = processing.signal( axes( indicator ), profiles{ index_image }{ index_options } );
                    else
                        profiles{ index_image }{ index_options } = processing.signal_matrix( axes( indicator ), profiles{ index_image }{ index_options } );
                    end

                    %------------------------------------------------------
                    % iii.) interpolate signals
                    %------------------------------------------------------
                    % TODO: add zeros and interpolate
%                     profiles{ index_image }{ index_options } = interpolate( profiles{ index_image }{ index_options }, options{ index_image }( index_options ).factor_interp );

                    % apply window function to smooth boundaries before DFT-based interpolation
    %                 profile_window = profile(:) .* tukeywin(N_samples, 0.1);

                end % for index_options = 1:numel( options{ index_image } )

            end % for index_image = 1:numel( images )

            % avoid cell array for single images
            if isscalar( images )
                profiles = profiles{ 1 };
            end

        end % function profiles = profile( images, options )

        %------------------------------------------------------------------
        % region extents
        %------------------------------------------------------------------
        function RBs = region_boundary( images, options )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % ensure class processing.image
            if ~isa( images, 'processing.image' )
                errorStruct.message = 'images must be processing.image!';
                errorStruct.identifier = 'profile:NoImages';
                error( errorStruct );
            end

            % ensure equal subclasses of math.grid_regular_orthogonal
            auxiliary.mustBeEqualSubclasses( 'math.grid_regular_orthogonal', images.grid );

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
            % 2.) compute regions
            %--------------------------------------------------------------
            % specify cell arrays
            N_samples = cell( size( images ) );
            volumes = cell( size( images ) );
            RBs = cell( size( images ) );

            % iterate image matrices
            for index_image = 1:numel( images )

                %----------------------------------------------------------
                % a) check arguments
                %----------------------------------------------------------
                % ensure class processing.options.region
                if ~isa( options{ index_image }, 'processing.options.region' )
                    errorStruct.message = sprintf( 'options{ %d } must be processing.options.region!', index_image );
                    errorStruct.identifier = 'region_boundary:NoOptions';
                    error( errorStruct );
                end

                %----------------------------------------------------------
                % b) compute regions
                %----------------------------------------------------------
                % extract axes
                axes = get_axes( images( index_image ).grid );

                % specify cell arrays
                N_samples{ index_image } = cell( size( options{ index_image } ) );
                volumes{ index_image } = cell( size( options{ index_image } ) );

                % iterate options
                for index_options = 1:numel( options{ index_image } )

                    % ensure valid numbers of dimensions
                    if options{ index_image }( index_options ).ROI.N_dimensions ~= images( index_image ).grid.N_dimensions
                        errorStruct.message = sprintf( 'Numbers of dimensions in options{ %d }( %d ).ROI and images( %d ).grid must equal!', index_image, index_options, index_image );
                        errorStruct.identifier = 'region_boundary:DimensionMismatch';
                        error( errorStruct );
                    end

                    % cut out axis
                    [ ~, indicators ] = cut_out( axes, [ options{ index_image }( index_options ).ROI.intervals.lb ]', [ options{ index_image }( index_options ).ROI.intervals.ub ]' );

                    % initialize results w/ zeros
                    N_samples{ index_image }{ index_options } = zeros( 1, images( index_image ).N_images );
                    volumes{ index_image }{ index_options } = zeros( 1, images( index_image ).N_images );

                    % iterate images
                    for index_col = 1:images( index_image ).N_images

                        % subsampling
                        samples = reshape( images( index_image ).samples( :, index_col ), images( index_image ).grid.N_points_axis );
                        samples = samples( indicators{ : } );

                        % logarithmic compression
                        samples_dB = illustration.dB( samples, 20 );
                        indicator = ( samples_dB >= options{ index_image }( index_options ).boundary_dB );

                        N_samples{ index_image }{ index_options }( index_col ) = sum( indicator( : ) );
                        volumes{ index_image }{ index_options }( index_col ) = N_samples{ index_image }{ index_options }( index_col ) * images( index_image ).grid.cell_ref.volume;

                    end % for index_col = 1:images( index_image ).N_images

                end % for index_options = 1:numel( options{ index_image } )

                % create structure
                RBs{ index_image } = struct( 'N_samples', N_samples{ index_image }, 'volume', volumes{ index_image } );

            end % for index_image = 1:numel( images )

            % avoid cell array for single images
            if isscalar( images )
                RBs = RBs{ 1 };
            end

        end % function RBs = region_boundary( images, options )

    end % methods

end % classdef image
