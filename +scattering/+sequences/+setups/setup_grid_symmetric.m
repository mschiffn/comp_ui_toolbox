%
% superclass for all symmetric pulse-echo measurement setups based on orthogonal regular grids
%
% author: Martin F. Schiffner
% date: 2019-08-22
% modified: 2019-08-22
%
classdef setup_grid_symmetric < scattering.sequences.setups.setup

	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	%% properties
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	properties (SetAccess = private)

        % independent properties
        N_points_per_pitch_axis ( 1, : ) double { mustBeInteger, mustBePositive } = 1

        % dependent properties
        indices_grid_FOV_shift ( :, : )	% indices of laterally shifted grid points

	end % properties

	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	%% methods
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	methods

        %------------------------------------------------------------------
        % constructor
        %------------------------------------------------------------------
        function objects = setup_grid_symmetric( xdc_arrays, homogeneous_fluids, FOVs, strs_name )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % superclass ensures class pulse_echo_measurements.homogeneous_fluid

            % ensure cell array for structs_elements
            

            % ensure class math.grid_regular_orthogonal for structs_elements
            

            % ensure class math.grid_regular_orthogonal for grids_FOV

            %--------------------------------------------------------------
            % 2.) create symmetric spatial discretizations based on orthogonal regular grids
            %--------------------------------------------------------------
            % constructor of superclass
            objects@scattering.sequences.setups.setup( xdc_arrays, homogeneous_fluids, FOVs, strs_name );

            %
            [ indicator_symmetry, N_points_per_pitch_axis ] = issymmetric( objects );
            if any( ~indicator_symmetry( : ) )
                errorStruct.message = 'setups must be symmetric!';
                errorStruct.identifier = 'setup_grid_symmetric:NoSetups';
                error( errorStruct );
            end

            % ensure cell array for N_points_per_pitch_axis
            if ~iscell( N_points_per_pitch_axis )
                N_points_per_pitch_axis = { N_points_per_pitch_axis };
            end

            % iterate symmetric pulse-echo measurement setups based on orthogonal regular grids
            for index_object = 1:numel( objects )

                % set independent properties
                objects( index_object ).N_points_per_pitch_axis = N_points_per_pitch_axis{ index_object };

                % lateral shifts of grid points for each array element
                objects( index_object ).indices_grid_FOV_shift = shift_lateral( objects( index_object ), ( 1:objects( index_object ).xdc_array.N_elements ) );

            end % for index_object = 1:numel( objects )

        end % function objects = setup_grid_symmetric( xdc_arrays, homogeneous_fluids, FOVs, strs_name )

        %------------------------------------------------------------------
        % lateral shift (TODO: check for correctness)
        %------------------------------------------------------------------
        function indices_grids_shift = shift_lateral( setups_grid_symmetric, indices_element, varargin )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % ensure class scattering.sequences.setups.setup_grid_symmetric
            if ~isa( setups_grid_symmetric, 'scattering.sequences.setups.setup_grid_symmetric' )
                errorStruct.message = 'setups_grid_symmetric must be scattering.sequences.setups.setup_grid_symmetric!';
                errorStruct.identifier = 'shift_lateral:AsymmetricSetups';
                error( errorStruct );
            end

            % ensure cell array for indices_element
            if ~iscell( indices_element )
                indices_element = { indices_element };
            end

            % ensure nonempty indices_grids
            if nargin >= 3 && ~isempty( varargin{ 1 } )
                indices_grids = varargin{ 1 };
            else
                indices_grids = cell( size( setups_grid_symmetric ) );
                for index_object = 1:numel( setups_grid_symmetric )
                    indices_grids{ index_object } = ( 1:setups_grid_symmetric( index_object ).FOV.shape.grid.N_points );
                end
            end

            % ensure cell array for indices_grids
            if ~iscell( indices_grids )
                indices_grids = { indices_grids };
            end

            % multiple setups_grid_symmetric / single indices_element
            if ~isscalar( setups_grid_symmetric ) && isscalar( indices_element )
                indices_element = repmat( indices_element, size( setups_grid_symmetric ) );
            end

            % single setups_grid_symmetric / multiple indices_element
            if isscalar( setups_grid_symmetric ) && ~isscalar( indices_element )
                setups_grid_symmetric = repmat( setups_grid_symmetric, size( indices_element ) );
            end

            % multiple setups_grid_symmetric / single indices_grids
            if ~isscalar( setups_grid_symmetric ) && isscalar( indices_grids )
                indices_grids = repmat( indices_grids, size( setups_grid_symmetric ) );
            end

            % ensure equal number of dimensions and sizes
            auxiliary.mustBeEqualSize( setups_grid_symmetric, indices_element, indices_grids );

            %--------------------------------------------------------------
            % 2.) shift grid positions on symmetric regular grids
            %--------------------------------------------------------------
            % specify cell array for indices_grids_shift
            indices_grids_shift = cell( size( setups_grid_symmetric ) );

            % iterate symmetric spatial discretizations
            for index_grid = 1:numel( setups_grid_symmetric )

                % ensure positive integers for indices_element{ index_grid }
                mustBeInteger( indices_element{ index_grid } );
                mustBePositive( indices_element{ index_grid } );

                % ensure that indices_element{ index_grid } do not exceed number of elements
                if any( indices_element{ index_grid } > setups_grid_symmetric( index_grid ).xdc_array.N_elements )
                    errorStruct.message = sprintf( 'indices_element{ %d } exceeds number of elements!', index_grid );
                    errorStruct.identifier = 'shift_lateral:InvalidIndices';
                    error( errorStruct );
                end

                % ensure positive integers for indices_grids{ index_grid }
% TODO: check in inverse index transform
                mustBeInteger( indices_grids{ index_grid } );
                mustBePositive( indices_grids{ index_grid } );

                % ensure that indices_grids{ index_grid } do not exceed number of grid points
                if any( indices_grids{ index_grid } > setups_grid_symmetric( index_grid ).FOV.shape.grid.N_points )
                    errorStruct.message = sprintf( 'indices_grids{ %d } exceeds number of grid points!', index_grid );
                    errorStruct.identifier = 'shift_lateral:InvalidIndices';
                    error( errorStruct );
                end

                % number of dimensions (total and lateral)
                N_dimensions = setups_grid_symmetric( index_grid ).FOV.shape.grid.N_dimensions;
                N_dimensions_lateral = N_dimensions - 1;

                % numbers of elements along each lateral coordinate axis
                N_elements_axis = setups_grid_symmetric( index_grid ).xdc_array.N_elements_axis( 1:N_dimensions_lateral )';

                % shift in grid points required for current array element
                indices_element_axis = inverse_index_transform( setups_grid_symmetric( index_grid ).xdc_array, indices_element{ index_grid }( : ) );
                N_points_shift_axis = ( indices_element_axis - 1 ) .* setups_grid_symmetric( index_grid ).N_points_per_pitch_axis;

                % subscripts of indices_grids{ index_grid }
                indices_grids_axis = inverse_index_transform( setups_grid_symmetric( index_grid ).FOV.shape.grid, indices_grids{ index_grid } );

                % numbers of selected elements and selected grid points
                N_elements_sel = numel( indices_element{ index_grid } );
                N_points_sel = numel( indices_grids{ index_grid } );

                % shift grid points laterally
                indices_grids_axis = repmat( reshape( indices_grids_axis, [ N_points_sel, 1, N_dimensions ] ), [ 1, N_elements_sel ] );
                N_points_shift_axis = repmat( reshape( [ N_points_shift_axis, zeros( N_elements_sel, 1 ) ], [ 1, N_elements_sel, N_dimensions ] ), [ N_points_sel, 1 ] );
                indices_grids_shift_axis = indices_grids_axis - N_points_shift_axis;

                % check mirroring
                indicator = indices_grids_shift_axis <= 0;
                if any( indicator( : ) )

                    % compute offset for shift:
                    % 1.) compute minimum number of grid points (GP) along each coordinate axis to ensure presence left of the center axis of the first element [ FOV_pos_x(1) <= XDC_pos_ctr_x(1) ]:
                    %     a) GP coincide with centroids of vibrating faces for
                    %           N_elements_axis:odd && N_points_axis:odd ||
                    %           N_elements_axis:even && N_points_axis:odd && N_points_per_pitch_axis:even ||
                    %           N_elements_axis:even && N_points_axis:even && N_points_per_pitch_axis:odd
                    %        => N_{lb} = ( N_elements_axis - 1 ) .* N_points_per_pitch_axis + 1
                    %	  b) GP do not coincide with centroids of vibrating faces for
                    %           N_elements_axis:odd && N_points_axis:even ||
                    %           N_elements_axis:even && N_points_axis:odd && N_points_per_pitch_axis:odd ||
                    %           N_elements_axis:even && N_points_axis:even && N_points_per_pitch_axis:even
                    %        => N_{lb} = ( N_elements_axis - 1 ) .* N_points_per_pitch_axis
                    % 2.) number of GP left of the center axis of the first element
                    %        N_{l} = 0.5 .* ( N_points_axis - N_{lb} )
                    % 3.) index of first element to be mirrored:
                    %       a) GP on axis [left and right of symmetry axis + center + 1]
                    %          2 N_{l} + 2 = N_points_axis - ( N_elements_axis - 1 ) .* N_points_per_pitch_axis + 1
                    %       b) GP off axis [left and right of symmetry axis + 1]
                    %          2 N_{l} + 1 = N_points_axis - ( N_elements_axis - 1 ) .* N_points_per_pitch_axis + 1
                    %       => identical equations
                    index_offset_axis = setups_grid_symmetric( index_grid ).FOV.shape.grid.N_points_axis( 1:N_dimensions_lateral ) - ( N_elements_axis - 1 ) .* setups_grid_symmetric( index_grid ).N_points_per_pitch_axis + 1;
                    index_offset_axis = repmat( reshape( [ index_offset_axis, 0 ], [ 1, 1, N_dimensions ] ), [ N_points_sel, N_elements_sel ] );

                    % mirror missing values
                    indices_grids_shift_axis( indicator ) = index_offset_axis( indicator ) - indices_grids_shift_axis( indicator );

                end % if any( indicator( : ) )

                % convert subscripts to linear indices
                indices_grids_shift_axis = reshape( indices_grids_shift_axis, [ N_points_sel * N_elements_sel, N_dimensions ] );
                indices_grids_shift{ index_grid } = reshape( forward_index_transform( setups_grid_symmetric( index_grid ).FOV.shape.grid, indices_grids_shift_axis ), [ N_points_sel, N_elements_sel ] );

            end % for index_grid = 1:numel( setups_grid_symmetric )

            % avoid cell array for single setups_grid_symmetric
            if isscalar( setups_grid_symmetric )
                indices_grids_shift = indices_grids_shift{ 1 };
            end

        end % function indices_grids_shift = shift_lateral( setups_grid_symmetric, indices_element, varargin )

	end % methods

end % classdef setup_grid_symmetric < scattering.sequences.setups.setup
