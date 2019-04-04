%
% superclass for all regular grids
%
% author: Martin F. Schiffner
% date: 2018-01-23
% modified: 2019-04-01
%
classdef grid_regular < discretizations.grid

	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	% properties
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	properties (SetAccess = private)

        % independent properties
        offset_axis ( 1, : ) physical_values.length     % arbitrary offset
        cell_ref ( 1, 1 ) math.parallelotope            % elementary cell
        N_points_axis ( 1, : ) double { mustBeInteger, mustBePositive, mustBeNonempty } = [ 128, 128 ]	 % numbers of grid points along each coordinate axis (1)

    end % properties

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % methods
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	methods

        %------------------------------------------------------------------
        % constructor
        %------------------------------------------------------------------
        function objects = grid_regular( offset_axis, cells_ref, N_points_axis )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % ensure cell array for offset_axis
            if ~iscell( offset_axis )
                offset_axis = { offset_axis };
            end

            % ensure class math.parallelotope
            if ~isa( cells_ref, 'math.parallelotope' )
                errorStruct.message     = 'cells_ref must be math.parallelotope!';
                errorStruct.identifier	= 'grid_regular:NoParallelotopes';
                error( errorStruct );
            end

            % ensure cell array for N_points_axis
            if ~iscell( N_points_axis )
                N_points_axis = { N_points_axis };
            end

            % ensure equal number of dimensions and sizes
            auxiliary.mustBeEqualSize( offset_axis, cells_ref, N_points_axis );

            %--------------------------------------------------------------
            % 2.) compute positions of the grid points
            %--------------------------------------------------------------
            positions = cell( size( offset_axis ) );

            for index_object = 1:numel( positions )

                N_dimensions = size( offset_axis{ index_object }, 2 );
                N_points = prod( N_points_axis{ index_object }, 2 );

                positions{ index_object } = physical_values.meter( zeros( N_points, N_dimensions ) );
            end

            %--------------------------------------------------------------
            % 3.) constructor of superclass
            %--------------------------------------------------------------
            objects@discretizations.grid( positions );

            %--------------------------------------------------------------
            % 4.) set independent properties
            %--------------------------------------------------------------
            for index_object = 1:numel( objects )

                % ensure class physical_values.length
                if ~isa( offset_axis{ index_object }, 'physical_values.length' )
                    errorStruct.message     = 'offset_axis must be physical_values.length!';
                    errorStruct.identifier	= 'grid_regular:NoLength';
                    error( errorStruct );
                end

                % ensure equal number of dimensions and sizes
                auxiliary.mustBeEqualSize( offset_axis{ index_object }, cells_ref( index_object ).edge_lengths, N_points_axis{ index_object } );

                % set independent properties
                objects( index_object ).offset_axis = offset_axis{ index_object };
                objects( index_object ).cell_ref = cells_ref( index_object );
                objects( index_object ).N_points_axis = N_points_axis{ index_object };

                % set dependent properties
                objects( index_object ).positions = compute_positions( objects( index_object ) );

            end % for index_object = 1:numel( offset_axis )

        end % function objects = grid_regular( offset_axis, cells_ref, N_points_axis )

        %------------------------------------------------------------------
        % compute discrete positions of the grid points along each axis
        %------------------------------------------------------------------
        function positions_axis = compute_positions_axis( grids_regular )

            %
            positions_axis = cell( 1, grids_regular.N_dimensions );

            %
            for index_dim = 1:grids_regular.N_dimensions

                indices_axis = (0:grids_regular.N_points_axis( index_dim ) - 1)';
                positions_axis{ index_dim } = repmat( grids_regular.offset_axis, [grids_regular.N_points_axis( index_dim ), 1] ) + indices_axis * grids_regular.delta_axis( index_dim ) * grids_regular.lattice_vectors( index_dim, : );
            end

        end % function positions_axis = compute_positions_axis( grids_regular )

        %------------------------------------------------------------------
        % compute discrete spatial frequencies along each axis
        %------------------------------------------------------------------
        function frequencies_axis = compute_frequencies_axis( grids_regular )

            indices_shift = ceil( grids_regular.N_points_axis / 2 );

            frequencies_axis = cell( 1, grids_regular.N_dimensions );

            for index_dim = 1:grids_regular.N_dimensions

                if index_dim < grids_regular.N_dimensions
                    % create FFTshifted axis
                    indices_axis = ( ( indices_shift( index_dim ) - grids_regular.N_points_axis( index_dim ) ):( indices_shift( index_dim ) - 1 ) )';
                else
                    % create normal axis
                    indices_axis = ( 0:( grids_regular.N_points_axis( end ) - 1 ) )';
                end
                frequencies_axis{ index_dim } = indices_axis * grids_regular.lattice_vectors( index_dim, : ) / ( grids_regular.N_points_axis( index_dim ) * grids_regular.delta_axis( index_dim ) );
            end

        end % function frequencies_axis = compute_frequencies_axis( grids_regular )

        %------------------------------------------------------------------
        % compute discrete positions of the grid points
        %------------------------------------------------------------------
        function positions = compute_positions( grids_regular )

            % create cell array
            positions = cell( size( grids_regular ) );

            % iterate regular grids
            for index_object = 1:numel( grids_regular )

                % calculate indices along each coordinate axis
                indices_lattice = ( 0:(grids_regular( index_object ).N_points - 1) )';
                indices_axis = inverse_index_transform( grids_regular( index_object ), indices_lattice );

                % compute Cartesian coordinates of grid points
                positions_rel = indices_axis * ( grids_regular( index_object ).cell_ref.edge_lengths .* grids_regular( index_object ).cell_ref.basis );
                positions{ index_object } = grids_regular( index_object ).offset_axis + positions_rel;

            end % for index_object = 1:numel( grids_regular )

            % avoid cell array for single regular grid
            if numel( grids_regular ) == 1
                positions = positions{ 1 };
            end

        end % function positions = compute_positions( grids_regular )

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

        end % function indices_axis = inverse_index_transform( obj, indices_lattice )

        %------------------------------------------------------------------
        % forward index transform
        %------------------------------------------------------------------
        function indices_lattice = forward_index_transform( obj, indices_axis )

            % factors for forward index calculation
            factors = ones( obj.N_dimensions, 1 );
            for index_prod = 1:(obj.N_dimensions - 1)
                factors( index_prod ) = prod( obj.N_points_axis( (index_prod + 1):end ), 2 );
            end

            % compute grid indices
            indices_lattice = indices_axis * factors;

        end % function indices_lattice = forward_index_transform( obj, indices_axis )

    end % methods

end % classdef grid_regular < discretizations.grid
