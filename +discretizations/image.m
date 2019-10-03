%
% superclass for all images
%
% author: Martin F. Schiffner
% date: 2019-09-10
% modified: 2019-10-03
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
            % ensure class discretizations.image
            if ~isa( images, 'discretizations.image' )
                errorStruct.message = 'images must be discretizations.image!';
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
        function profiles = profile( images, dim, varargin )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % ensure class discretizations.image
            if ~isa( images, 'discretizations.image' )
                errorStruct.message = 'images must be discretizations.image!';
                errorStruct.identifier = 'profile:NoImages';
                error( errorStruct );
            end

            % ensure nonempty positive integers
            mustBeNonempty( dim );
            mustBePositive( dim );
            mustBeInteger( dim );

            % ensure nonempty intervals_pos
            if nargin >= 3 && ~isempty( varargin{ 1 } )
                intervals_pos = varargin{ 1 };
            else
                intervals_pos = physical_values.meter( zeros( size( images ) ) );
                for index_image = 1:numel( images )
                    images( index_image ).grid.positions;
                end
            end

            % ensure class math.interval
            if ~isa( intervals_pos, 'math.interval' )
                errorStruct.message = 'intervals_pos must be math.interval!';
                errorStruct.identifier = 'profile:NoIntervals';
                error( errorStruct );
            end

            %
            auxiliary.mustBeEqualSubclasses( 'physical_values.length', intervals_pos.lb );

            % ensure nonempty N_zeros_add
            if nargin >= 4 && ~isempty( varargin{ 2 } )
                N_zeros_add = varargin{ 2 };
            else
                N_zeros_add = 50;
            end

            % ensure nonempty factor_interp
            if nargin >= 5 && ~isempty( varargin{ 3 } )
                factor_interp = varargin{ 3 };
            else
                factor_interp = 10;
            end

            % single images / multiple intervals_pos
            if isscalar( images ) && ~isscalar( intervals_pos )
                images = repmat( images, size( intervals_pos ) );
            end

            % multiple images / single dim
            if ~isscalar( images ) && isscalar( dim )
                dim = repmat( dim, size( images ) );
            end

            % multiple images / single intervals_pos
            if ~isscalar( images ) && isscalar( intervals_pos )
                intervals_pos = repmat( intervals_pos, size( images ) );
            end

            % multiple images / single N_zeros_add
            if ~isscalar( images ) && isscalar( N_zeros_add )
                N_zeros_add = repmat( N_zeros_add, size( images ) );
            end

            % multiple images / single factor_interp
            if ~isscalar( images ) && isscalar( factor_interp )
                factor_interp = repmat( factor_interp, size( images ) );
            end

            % ensure equal number of dimensions and sizes
            auxiliary.mustBeEqualSize( images, dim, intervals_pos, N_zeros_add, factor_interp );

            %--------------------------------------------------------------
            % 2.) compute profiles
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

                %
                mustBeLessThanOrEqual( dim( index_image ), images( index_image ).grid.N_dimensions );

                %----------------------------------------------------------
                % b) compute profiles in specified window
                %----------------------------------------------------------
                axes = get_axes( images( index_image ).grid );
                N_points_axis_act = images( index_image ).grid.N_points_axis;

                % cut out axis
                [ ~, indicator ] = cut_out( axes( dim( index_image ) ), intervals_pos( index_image ).lb, intervals_pos( index_image ).ub );

                %
                str_selector = repmat( { ':' }, [ 1, images( index_image ).grid.N_dimensions ] );
                str_selector{ dim( index_image ) } = indicator;

                % specify cell array for profiles
                profiles{ index_image } = cell( 1, images( index_image ).N_images );

                % iterate images
                for index_col = 1:images( index_image ).N_images

                    samples_act = reshape( images( index_image ).samples( :, index_col ), images( index_image ).grid.N_points_axis );
                    profiles{ index_image }{ index_col } = squeeze( vecnorm( samples_act( str_selector{ : } ), 2, dim( index_image ) ) ) * sqrt( images( index_image ).grid.cell_ref.edge_lengths( dim( index_image ) ) );

                end % for index_col = 1:images( index_image ).N_images

%                 indicator = ( images( index_image ).grid.positions( :, dim( index_image ) ) >= intervals_pos( index_image ).lb ) & ( images( index_image ).grid.positions( :, dim( index_image ) ) <= intervals_pos( index_image ).ub );
%                 N_points_axis_act( dim( index_image ) ) = sum( indicator ) * N_points_axis_act( dim( index_image ) ) / images( index_image ).grid.N_points;

                %----------------------------------------------------------
                % c) create signal matrices
                %----------------------------------------------------------
% TODO: check number of dimensions
                
                N_points_axis_act( dim( index_image ) ) = 1;

                indicator = N_points_axis_act > 1;
                if images( index_image ).N_images == 1
                    profiles{ index_image } = discretizations.signal( axes( indicator ), profiles{ index_image } );
                else
                    profiles{ index_image } = discretizations.signal_matrix( axes( indicator ), profiles{ index_image } );
                end

                %----------------------------------------------------------
                % d) interpolate signals
                %----------------------------------------------------------
%                 profiles{ index_image } = interpolate( profiles{ index_image }, factor_interp( index_image ) );

                % apply window function to smooth boundaries before DFT-based interpolation
%                 profile_window = profile(:) .* tukeywin(N_samples, 0.1);

                % add zeros and interpolate

            end % for index_image = 1:numel( images )

            % avoid cell array for single images
            if isscalar( images )
                profiles = profiles{ 1 };
            end

        end % function profiles = profile( images, dim, varargin )

        %------------------------------------------------------------------
        % region
        %------------------------------------------------------------------
        function RBs = region_boundary( images, boundaries_dB, varargin )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % ensure class discretizations.image
            if ~isa( images, 'discretizations.image' )
                errorStruct.message = 'images must be discretizations.image!';
                errorStruct.identifier = 'profile:NoImages';
                error( errorStruct );
            end

            % ensure equal subclasses of math.grid_regular_orthogonal
            auxiliary.mustBeEqualSubclasses( 'math.grid_regular_orthogonal', images.grid );

            % ensure nonempty ROIs
            if nargin >= 3 && ~isempty( varargin{ 1 } )
                ROIs = varargin{ 1 };
            else
                ROIs = cell( numel( images ), 1 );
                for index_image = 1:numel( images )
                    lbs = images( index_image ).grid.offset_axis - 0.5 .* images( index_image ).grid.cell_ref.edge_lengths;
                    ubs = images( index_image ).grid.offset_axis + ( images( index_image ).grid.N_points_axis - 0.5 ) .* images( index_image ).grid.cell_ref.edge_lengths;
                    intervals = num2cell( math.interval( lbs, ubs ) );
                    ROIs{ index_image } = math.orthotope( intervals{ : } );
                end
                ROIs = reshape( cat( 1, ROIs{ : } ), size( images ) );
            end

            % ensure class math.orthotope
            if ~isa( ROIs, 'math.orthotope' )
                errorStruct.message = 'ROIs must be math.orthotope!';
                errorStruct.identifier = 'region_boundary:NoOrthotopes';
                error( errorStruct );
            end

            % single images / multiple ROIs
            if isscalar( images ) && ~isscalar( ROIs )
                images = repmat( images, size( ROIs ) );
            end

            % multiple images / single boundaries_dB
            if ~isscalar( images ) && isscalar( boundaries_dB )
                boundaries_dB = repmat( boundaries_dB, size( images ) );
            end

            % multiple images / single ROIs
            if ~isscalar( images ) && isscalar( ROIs )
                ROIs = repmat( ROIs, size( images ) );
            end

            % ensure equal number of dimensions and sizes
            auxiliary.mustBeEqualSize( images, boundaries_dB, ROIs );

            %--------------------------------------------------------------
            % 2.) compute regions
            %--------------------------------------------------------------
            % specify cell arrays
            N_samples = cell( size( images ) );
            volumes = cell( size( images ) );

            % iterate image matrices
            for index_image = 1:numel( images )

                %----------------------------------------------------------
                % a) check arguments
                %----------------------------------------------------------
                % ensure class math.grid_regular_orthogonal
                if ~isa( images( index_image ).grid, 'math.grid_regular_orthogonal' )
                    errorStruct.message = sprintf( 'images( %d ).grid must be math.grid_regular_orthogonal!', index_image );
                    errorStruct.identifier = 'region_boundary:NoOrthogonalRegularGrid';
                    error( errorStruct );
                end

                % ensure valid numbers of dimensions
                if ROIs( index_image ).N_dimensions ~= images( index_image ).grid.N_dimensions
                    errorStruct.message = sprintf( 'Numbers of dimensions in ROIs( %d ) and images( %d ).grid must equal!', index_image, index_image );
                    errorStruct.identifier = 'region_boundary:DimensionMismatch';
                    error( errorStruct );
                end

                %----------------------------------------------------------
                % b)
                %----------------------------------------------------------
                % cut out axes
                axes = get_axes( images( index_image ).grid );
                [ ~, indicators ] = cut_out( axes, [ ROIs( index_image ).intervals.lb ]', [ ROIs( index_image ).intervals.ub ]' );

                N_samples{ index_image } = zeros( 1, images( index_image ).N_images );
                volumes{ index_image } = zeros( 1, images( index_image ).N_images );

                % iterate images
                for index_col = 1:images( index_image ).N_images

                    % subsampling
                    samples = reshape( images( index_image ).samples( :, index_col ), images( index_image ).grid.N_points_axis );
                    samples = samples( indicators{ : } );

                    % logarithmic compression
                    samples_dB = illustration.dB( samples, 20 );
                    indicator = ( samples_dB >= boundaries_dB( index_image ) );

                    N_samples{ index_image }( index_col ) = sum( indicator( : ) );
                    volumes{ index_image }( index_col ) = N_samples{ index_image }( index_col ) * images( index_image ).grid.cell_ref.volume;

                end % for index_col = 1:images( index_image ).N_images

            end % for index_image = 1:numel( images )

            % create structure
            RBs = struct( 'N_samples', N_samples, 'volume', volumes );

        end % function RBs = region_boundary( images, boundaries_dB, varargin )

    end % methods

end % classdef image
