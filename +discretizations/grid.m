%
% superclass for all regular grids
%
% author: Martin F. Schiffner
% date: 2018-01-23
% modified: 2019-03-18
%
classdef grid

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	% properties
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	properties (SetAccess = private)

        % independent properties
        N_dimensions ( 1, 1 ) double { mustBeInteger, mustBePositive, mustBeNonempty } = 2              % number of dimensions (1)
        N_points_axis ( 1, : ) double { mustBeInteger, mustBePositive, mustBeNonempty } = [ 128, 128 ]	% numbers of grid points along each coordinate axis (1)
        delta_axis ( 1, : ) double { mustBeReal, mustBePositive, mustBeNonempty } = [ 1e-4, 1e-4 ]      % constant spacings between the adjacent grid points along each coordinate axis (m)
        offset_axis ( 1, : ) double { mustBeReal, mustBeNonempty } = [ -63.5e-4, 0.5e-4]                % arbitrary offset (m)
        lattice_vectors ( :, : ) double { mustBeReal, mustBeNonempty } = eye( 2 )                       % linearly-independent unit row vectors of the grid (m)
        str_name                                                                                        % name of the grid

        % dependent properties
        N_points ( 1, 1 ) double { mustBeInteger, mustBePositive, mustBeNonempty } = 16384	% total number of grid points (1)
        delta_V ( 1, 1 ) double { mustBeReal, mustBePositive, mustBeNonempty } = 1e-8       % d-dimensional volume element (m^{d})
        positions_axis      % discrete positions of the grid points along each axis (m)
        positions           % discrete positions of the grid points (m)
        frequencies_axis	% discrete spatial frequencies along each axis (1 / m)
    end % properties

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % methods
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    methods

        %------------------------------------------------------------------
        % constructor
        %------------------------------------------------------------------
        function objects = grid( N_points_axis, delta_axis, offset_axis, varargin )

            % return for no input arguments
            if nargin == 0
                return;
            end

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % ensure cell array
            if ~iscell( N_points_axis )
                N_points_axis = { N_points_axis };
            end

            % ensure cell array
            if ~iscell( delta_axis )
                delta_axis = { delta_axis };
            end

            % ensure cell array
            if ~iscell( offset_axis )
                offset_axis = { offset_axis };
            end

            % ensure equal number of dimensions and sizes
            auxiliary.mustBeEqualSize( N_points_axis, delta_axis, offset_axis );
            % assertion: N_points_axis, delta_axis, and offset_axis have the same size

            %--------------------------------------------------------------
            % 2.) create regular grids
            %--------------------------------------------------------------
            N_objects = numel( N_points_axis );
            objects = repmat( objects, size( N_points_axis ) );

            % check and set independent properties
            for index_object = 1:N_objects

                % ensure equal number of dimensions and sizes
                auxiliary.mustBeEqualSize( N_points_axis{ index_object }, delta_axis{ index_object }, offset_axis{ index_object } );
                % assertion: N_points_axis, delta_axis, and offset_axis have the same size

                % set independent properties
                objects( index_object ).N_dimensions = numel( N_points_axis{ index_object } );
                objects( index_object ).N_points_axis = N_points_axis{ index_object };
                objects( index_object ).delta_axis = delta_axis{ index_object };
                objects( index_object ).offset_axis = offset_axis{ index_object };
                % assertion: independent properties specify valid grid

                % optional basis unit vectors of the grid
                if numel( varargin ) > 0
                    if abs( det( varargin{ 1 } ) ) > eps
                        objects( index_object ).lattice_vectors = varargin{ 1 };
                    else
                        % TODO: invalid grid directions
                    end
                else
                    % specify canonical basis
                    objects( index_object ).lattice_vectors = eye( objects( index_object ).N_dimensions );
                end

                % dependent properties
                objects( index_object ).N_points = prod( objects( index_object ).N_points_axis, 2 );
                % TODO: check volume of n-parallelotope
                objects( index_object ).delta_V = abs( det( diag( objects( index_object ).delta_axis ) * objects( index_object ).lattice_vectors ) );

                % compute discrete positions of the grid points along each axis
                objects( index_object ) = compute_positions_axis( objects( index_object ) );

                % compute discrete positions of the grid points
                objects( index_object ) = compute_positions( objects( index_object ) );

                % compute discrete spatial frequencies along each axis
                objects( index_object ) = compute_frequencies_axis( objects( index_object ) );

            end % for index_object = 1:N_objects

        end % function objects = grid( N_points_axis, delta_axis, offset_axis, varargin )

        %------------------------------------------------------------------
        % compute discrete positions of the grid points along each axis
        %------------------------------------------------------------------
        function obj = compute_positions_axis( obj )

            obj.positions_axis = cell( 1, obj.N_dimensions );
            for index_dim = 1:obj.N_dimensions

                indices_axis = (0:obj.N_points_axis( index_dim ) - 1)';
                obj.positions_axis{ index_dim } = repmat( obj.offset_axis, [obj.N_points_axis( index_dim ), 1] ) + indices_axis * obj.delta_axis( index_dim ) * obj.lattice_vectors( index_dim, : );
            end
        end

        %------------------------------------------------------------------
        % compute discrete spatial frequencies along each axis
        %------------------------------------------------------------------
        function obj = compute_frequencies_axis( obj )

            indices_shift = ceil( obj.N_points_axis / 2 );

            obj.frequencies_axis = cell( 1, obj.N_dimensions );
            for index_dim = 1:obj.N_dimensions

                if index_dim < obj.N_dimensions
                    % create FFTshifted axis
                    indices_axis = ( ( indices_shift( index_dim ) - obj.N_points_axis( index_dim ) ):( indices_shift( index_dim ) - 1 ) )';
                else
                    % create normal axis
                    indices_axis = ( 0:( obj.N_points_axis( end ) - 1 ) )';
                end
                obj.frequencies_axis{ index_dim } = indices_axis * obj.lattice_vectors( index_dim, : ) / ( obj.N_points_axis( index_dim ) * obj.delta_axis( index_dim ) );
            end
        end

        %------------------------------------------------------------------
        % compute discrete positions of the grid points
        %------------------------------------------------------------------
        function obj = compute_positions( obj )

            % calculate indices along each coordinate axis
            indices_lattice = (0:(obj.N_points - 1))';
            indices_axis = obj.inverse_index_transform( indices_lattice );

            % compute positions
            obj.positions = repmat( obj.offset_axis, [obj.N_points, 1] ) + ( indices_axis .* repmat( obj.delta_axis, [obj.N_points, 1] ) ) * obj.lattice_vectors;
        end

        %------------------------------------------------------------------
        % inverse index transform
        %------------------------------------------------------------------
        function indices_axis = inverse_index_transform( obj, indices_lattice )

            % divisors for inverse index calculation
            divisors = zeros( 1, obj.N_dimensions - 1 );
            for index_prod = 2:obj.N_dimensions
                divisors( index_prod - 1 ) = prod( obj.N_points_axis( index_prod:end ), 2 );
            end

            % compute indices along each coordinate axis
            indices_axis = zeros( obj.N_points, obj.N_dimensions );

            for index_dimension = 1:(obj.N_dimensions - 1)
                indices_axis( :, index_dimension ) = floor( indices_lattice(:) / divisors( index_dimension ) );
                indices_lattice(:) = indices_lattice(:) - indices_axis( :, index_dimension ) * divisors( index_dimension );
            end
            indices_axis( :, obj.N_dimensions ) = indices_lattice;
        end

        %------------------------------------------------------------------
        % forward index transform
        %------------------------------------------------------------------
        function indices_lattice = forward_index_transform( obj, indices_axis )

            % factors for forward index calculation
            factors = zeros( obj.N_dimensions, 1 );
            for index_prod = 1:(obj.N_dimensions - 1)
                factors( index_prod ) = prod( obj.N_points_axis( (index_prod + 1):end ), 2 );
            end
            factors(end) = 1;

            % compute grid indices
            indices_lattice = indices_axis * factors;
        end

        %------------------------------------------------------------------
        % mutual differences
        %------------------------------------------------------------------
        function differences = mutual_differences( grids_1, grids_2, varargin )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % ensure class discretizations.grid
            if ~( isa( grids_1, 'discretizations.grid' ) && isa( grids_2, 'discretizations.grid' ) )
                errorStruct.message     = 'Both arguments must be discretizations.grid!';
                errorStruct.identifier	= 'mutual_differences:NoGrids';
                error( errorStruct );
            end

            % ensure nonempty indices_1
            if nargin >= 3 && ~isempty( varargin{ 1 } )
                indices_1 = varargin{ 1 };
            else
                indices_1 = cell( size( grids_1 ) );
                for index_object = 1:numel( grids_1 )
                    indices_1{ index_object } = (1:grids_1( index_object ).N_points);
                end
            end

            % ensure cell array for indices_1
            if ~iscell( indices_1 )
                indices_1 = { indices_1 };
            end

            % ensure nonempty indices_2
            if nargin >= 4 && ~isempty( varargin{ 2 } )
                indices_2 = varargin{ 2 };
            else
                indices_2 = cell( size( grids_2 ) );
                for index_object = 1:numel( grids_2 )
                    indices_2{ index_object } = (1:grids_2( index_object ).N_points);
                end
            end

            % ensure cell array for indices_2
            if ~iscell( indices_2 )
                indices_2 = { indices_2 };
            end

            % multiple grids_1 / single indices_1
            if ~isscalar( grids_1 ) && isscalar( indices_1)
                indices_1 = repmat( indices_1, size( grids_1 ) );
            end

            % multiple grids_2 / single indices_2
            if ~isscalar( grids_2 ) && isscalar( indices_2)
                indices_2 = repmat( indices_2, size( grids_2 ) );
            end

            % ensure equal number of dimensions and sizes
            auxiliary.mustBeEqualSize( grids_1, grids_2, indices_1, indices_2 );

            %--------------------------------------------------------------
            % 2.) compute mutual differences for each pair of grids
            %--------------------------------------------------------------
            % specify cell array
            differences = cell( size( grids_1 ) );

            for index_object = 1:numel( grids_1 )

                % maximum number of dimensions
                N_dimensions_max = max( [ grids_1( index_object ).N_dimensions, grids_2( index_object ).N_dimensions ] );

                % numbers of relevant grid points
                N_points_1 = numel( indices_1{ index_object } );
                N_points_2 = numel( indices_2{ index_object } );

                % inflate relevant positions to correct dimension
                positions_1 = [ grids_1( index_object ).positions( indices_1{ index_object }, : ), zeros( N_points_1, N_dimensions_max - grids_1( index_object ).N_dimensions ) ];
                positions_2 = [ grids_2( index_object ).positions( indices_2{ index_object }, : ), zeros( N_points_2, N_dimensions_max - grids_2( index_object ).N_dimensions ) ];

                % reshape relevant positions to compute mutual distances
                positions_1 = repmat( reshape( positions_1, [ N_points_1, 1, N_dimensions_max ] ), [ 1, N_points_2, 1] );
                positions_2 = repmat( reshape( positions_2, [ 1, N_points_2, N_dimensions_max ] ), [ N_points_1, 1, 1] );

                % compute differences
                differences{ index_object } = positions_1 - positions_2;

            end % for index_object = 1:numel( grids_1 )

            % avoid cell array for single pair of grids
            if numel( grids_1 ) == 1
                differences = differences{ 1 };
            end

        end % function differences = mutual_differences( grids_1, grids_2, varargin )

        %------------------------------------------------------------------
        % mutual distances
        %------------------------------------------------------------------
        function [ D, differences ] = mutual_distances( grids_1, grids_2, varargin )

            %--------------------------------------------------------------
            % 1.) mutual differences
            %--------------------------------------------------------------
            differences = mutual_differences( grids_1, grids_2, varargin{ : } );

            % ensure cell array
            if ~iscell( differences )
                differences = { differences };
            end

            %--------------------------------------------------------------
            % 2.) compute mutual distances for each pair of grids
            %--------------------------------------------------------------
            D = cell( size( differences ) );

            for index_object = 1:numel( differences )

                % compute l2-norms
                D{ index_object } = sqrt( sum( differences{ index_object }.^2, 3 ) );

            end % for index_object = 1:numel( differences )

            % avoid cell array for single pair of grids
            if numel( grids_1 ) == 1
                D = D{ 1 };
                differences = differences{ 1 };
            end

        end % function [ D, differences ] = mutual_distances( grids_1, grids_2, varargin )

        %------------------------------------------------------------------
        % mutual unit vectors
        %------------------------------------------------------------------
        function [ e_1_minus_2, D ] = mutual_unit_vectors( grids_1, grids_2, varargin )

            %--------------------------------------------------------------
            % 1.) mutual distances
            %--------------------------------------------------------------
            [ D, differences ] = mutual_distances( grids_1, grids_2, varargin{ : } );

            % ensure cell array for D
            if ~iscell( D )
                D = { D };
            end

            % ensure cell array for differences
            if ~iscell( differences )
                differences = { differences };
            end

            %--------------------------------------------------------------
            % 2.) compute mutual unit vectors for each pair of grids
            %--------------------------------------------------------------
            e_1_minus_2 = cell( size( differences ) );

            for index_object = 1:numel( differences )

                % compute unit vectors
                e_1_minus_2{ index_object } = differences{ index_object } ./ D{ index_object };

            end % for index_object = 1:numel( differences )

            % avoid cell array for single pair of grids
            if numel( grids_1 ) == 1
                D = D{ 1 };
                e_1_minus_2 = e_1_minus_2{ 1 };
            end

        end % function [ e_1_minus_2, D ] = mutual_unit_vectors( grids_1, grids_2, varargin )

    end % methods

end % classdef grid
