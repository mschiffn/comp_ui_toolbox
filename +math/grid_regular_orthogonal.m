%
% superclass for all orthogonal regular grids
%
% author: Martin F. Schiffner
% date: 2018-01-23
% modified: 2022-04-07
%
classdef grid_regular_orthogonal < math.grid_regular

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % methods
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	methods

        %------------------------------------------------------------------
        % constructor
        %------------------------------------------------------------------
        function objects = grid_regular_orthogonal( offset_axis, delta_axis, N_points_axis )

            %--------------------------------------------------------------
            % 1.) create cuboid reference cells
            %--------------------------------------------------------------
            cells_ref = math.cuboid( delta_axis );

            %--------------------------------------------------------------
            % 2.) constructor of superclass
            %--------------------------------------------------------------
            objects@math.grid_regular( offset_axis, cells_ref, N_points_axis );

        end % function objects = grid_regular_orthogonal( offset_axis, delta_axis, N_points_axis )

        %------------------------------------------------------------------
        % compute discrete positions of the grid points along each axis
        %------------------------------------------------------------------
        function axes = get_axes( grids_regular_orthogonal )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % ensure class math.grid_regular
            if ~isa( grids_regular_orthogonal, 'math.grid_regular' )
                errorStruct.message = 'grids_regular_orthogonal must be math.grid_regular!';
                errorStruct.identifier = 'get_axes:NoOrthogonalRegularGrids';
                error( errorStruct );
            end

            %--------------------------------------------------------------
            % 2.) extract axes
            %--------------------------------------------------------------
            % specify cell array for axes
            axes = cell( size( grids_regular_orthogonal ) );

            % iterate regular grids
            for index_object = 1:numel( grids_regular_orthogonal )

                % create increasing sequences w/ regular spacing
                axes{ index_object } = math.sequence_increasing_regular( grids_regular_orthogonal( index_object ).offset_axis, grids_regular_orthogonal( index_object ).cell_ref.edge_lengths, grids_regular_orthogonal( index_object ).N_points_axis );

            end % for index_object = 1:numel( grids_regular_orthogonal )

            % avoid cell array for single grids_regular_orthogonal
            if isscalar( grids_regular_orthogonal )
                axes = axes{ 1 };
            end

        end % function axes = get_axes( grids_regular_orthogonal )

        %------------------------------------------------------------------
        % compute discrete spatial frequencies along each axis
        %------------------------------------------------------------------
        function axes = get_frequency_axes( grids_regular_orthogonal )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % ensure class math.grid_regular_orthogonal
            if ~isa( grids_regular_orthogonal, 'math.grid_regular_orthogonal' )
                errorStruct.message = 'grids_regular_orthogonal must be math.grid_regular_orthogonal!';
                errorStruct.identifier = 'get_frequency_axes:NoOrthogonalRegularGrids';
                error( errorStruct );
            end

            %--------------------------------------------------------------
            % 2.) compute discrete spatial frequencies
            %--------------------------------------------------------------
            % specify cell array for axes
            axes = cell( size( grids_regular_orthogonal ) );

            % iterate regular grids
            for index_object = 1:numel( grids_regular_orthogonal )

                % shift indices
                indices_shift = ceil( grids_regular_orthogonal( index_object ).N_points_axis / 2 );
                indices_shift( end ) = grids_regular_orthogonal( index_object ).N_points_axis( end );

                % lower and upper bounds
                lbs_q = indices_shift - grids_regular_orthogonal( index_object ).N_points_axis;
                ubs_q = lbs_q + grids_regular_orthogonal( index_object ).N_points_axis - 1;

                % regular spacings
                deltas = 1 ./ ( grids_regular_orthogonal( index_object ).N_points_axis .* grids_regular_orthogonal( index_object ).cell_ref.edge_lengths );

                % create increasing sequences w/ regular spacing and quantized bounds
                axes{ index_object } = math.sequence_increasing_regular_quantized( lbs_q, ubs_q, deltas );

            end % for index_object = 1:numel( grids_regular_orthogonal )

            % avoid cell array for single grids_regular_orthogonal
            if isscalar( grids_regular_orthogonal )
                axes = axes{ 1 };
            end

        end % function axes = get_frequency_axes( grids_regular_orthogonal )

	end % methods

end % classdef grid_regular_orthogonal < math.grid_regular
