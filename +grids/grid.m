%
% superclass for all regular grids
%
% author: Martin F. Schiffner
% date: 2018-01-23
% modified: 2019-01-17
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
        function obj = grid( N_points_axis, delta_axis, offset_axis, varargin )

            % return for no input arguments
            if nargin == 0
                return;
            end

            % check and set independent properties
            % TODO: check number of dimensions
            obj.N_dimensions = numel( N_points_axis );
            obj.N_points_axis = N_points_axis;
            obj.delta_axis = delta_axis;
            obj.offset_axis = offset_axis;
            % assertion: independent properties specify valid grid

            % optional basis unit vectors of the grid
            if numel( varargin ) > 0
                if abs( det( varargin{ 1 } ) ) > eps
                    obj.lattice_vectors = varargin{ 1 };
                else
                    % TODO: invalid grid directions
                end
            else
                % specify canonical basis
                obj.lattice_vectors = eye( obj.N_dimensions );
            end

            % dependent properties
            obj.N_points = prod( obj.N_points_axis, 2 );
            obj.delta_V  = abs( det( diag( obj.delta_axis ) * obj.lattice_vectors ) ); % TODO: check volume of n-parallelotope

            % compute discrete positions of the grid points along each axis
            obj = compute_positions_axis( obj );

            % compute discrete positions of the grid points
            obj = compute_positions( obj );

            % compute discrete spatial frequencies along each axis
            obj = compute_frequencies_axis( obj );
        end

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
    end % methods

end % classdef grid
