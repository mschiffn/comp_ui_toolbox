%
% superclass for all symmetric spatial discretizations based on orthogonal regular grids
%
% author: Martin F. Schiffner
% date: 2019-02-17
% modified: 2019-05-05
%
classdef spatial_grid_symmetric < discretizations.spatial_grid

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	%% properties
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	properties (SetAccess = private)

        % independent properties
        N_points_per_pitch_axis ( 1, : ) double { mustBeInteger, mustBePositive } = 1

	end % properties

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% methods
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	methods

        %------------------------------------------------------------------
        % constructor
        %------------------------------------------------------------------
        function objects = spatial_grid_symmetric( grids_elements, grids_FOV, N_points_per_pitch_axis )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % ensure cell array for grids_elements
            if ~iscell( grids_elements )
                grids_elements = { grids_elements };
            end

            % ensure class discretizations.grid_regular_orthogonal for grids_elements
            for index_object = 1:numel( grids_elements )
                if ~isa( [grids_elements{ index_object }.grid], 'discretizations.grid_regular_orthogonal' )
                    errorStruct.message = sprintf( 'grids_elements{ %d } must be discretizations.grid_regular_orthogonal!', index_object );
                    errorStruct.identifier = 'spatial_grid_symmetric:NoRegularOrthogonalGrid';
                    error( errorStruct );
                end
            end % for index_object = 1:numel( grids_elements )

            % ensure class discretizations.grid_regular_orthogonal for grids_FOV
            if ~isa( grids_FOV, 'discretizations.grid_regular_orthogonal' )
                errorStruct.message = 'grids_FOV must be discretizations.grid_regular_orthogonal!';
                errorStruct.identifier = 'spatial_grid_symmetric:NoRegularOrthogonalGrid';
                error( errorStruct );
            end

            % ensure cell array for N_points_per_pitch_axis
            if ~iscell( N_points_per_pitch_axis )
                N_points_per_pitch_axis = { N_points_per_pitch_axis };
            end

            %--------------------------------------------------------------
            % 2.) constructor of superclass
            %--------------------------------------------------------------
            objects@discretizations.spatial_grid( grids_elements, grids_FOV );

            %--------------------------------------------------------------
            % 3.) confirm symmetry
            %--------------------------------------------------------------
            for index_object = 1:numel( objects )
% TODO: symmetry of apodization weights?
                %----------------------------------------------------------
                % a) check lateral symmetry of FOV about the axial axis
                %----------------------------------------------------------
                FOV_pos_ctr = 2 * grids_FOV( index_object ).offset_axis( 1:(end - 1) ) + ( grids_FOV( index_object ).N_points_axis( 1:(end - 1) ) - 1 ) .* grids_FOV( index_object ).cell_ref.edge_lengths( 1:(end - 1) );
                if ~all( abs( double( FOV_pos_ctr ) ) < eps )
                    errorStruct.message     = 'Symmetric spatial grid requires the symmetry of FOV about the axial axis!';
                    errorStruct.identifier	= 'spatial_grid_symmetric:NoSymmetry';
                    error( errorStruct );
                end
%TODO: check minimal # of lateral grid points
                %----------------------------------------------------------
                % b) lateral spacing is an integer fraction of the element pitch
                %    => translational invariance by shifts of factor_interp_tx points
                %----------------------------------------------------------
                % ensure positive intergers for N_points_per_pitch_axis{ index_object }
                mustBeInteger( N_points_per_pitch_axis{ index_object } );
                mustBePositive( N_points_per_pitch_axis{ index_object } );

%                 % compute mutual distances of grid points
%                 D = mutual_distances( grids_elements{ index_object }( 1:(end - 1) ), grids_elements{ index_object }( 2:end ) );
% 
%                 % ensure cell array for D
%                 if ~iscell( D )
%                     D = { D };
%                 end
% 
%                 % check diagonals
%                 for index_D = 1:numel( D )
%                     mean( diag( D{ index_D } ) )
%                 end
% 
%                 N_points_per_pitch = ( grids_elements{ index_object }( 2 ).positions( 1 ) - grids_elements{ index_object }( 1 ).positions( 1 ) ) ./ grids_FOV( index_object ).cell_ref.edge_lengths( 1:(end - 1) );
%                 % TODO: why is the error so large?
%                 if ~all( abs( N_points_per_pitch - floor( N_points_per_pitch ) ) < floor( N_points_per_pitch ) / 10^12 )
%                     errorStruct.message     = 'Symmetric spatial grid requires the lateral spacings of the grid points in the FOV to be an integer fraction of the element pitch!';
%                     errorStruct.identifier	= 'spatial_grid_symmetric:NoIntegerFraction';
%                     error( errorStruct );
%                 end

                % set independent properties
                objects( index_object ).N_points_per_pitch_axis = N_points_per_pitch_axis{ index_object };

            end % for index_object = 1:numel( objects )

        end % function objects = spatial_grid_symmetric( grids_elements, grids_FOV, N_points_per_pitch_axis )

        %------------------------------------------------------------------
        % lateral shift (TODO: check for correctness)
        %------------------------------------------------------------------
        function indices_grids_shift = shift_lateral( spatial_grids_symmetric, indices_element, varargin )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % ensure cell array for indices_element
            if ~iscell( indices_element )
                indices_element = { indices_element };
            end

            % ensure nonempty indices_grids
            if nargin >= 3 && ~isempty( varargin{ 1 } )
                indices_grids = varargin{ 1 };
            else
                indices_grids = cell( size( spatial_grids_symmetric ) );
                for index_object = 1:numel( spatial_grids_symmetric )
                    indices_grids{ index_object } = ( 1:spatial_grids_symmetric( index_object ).grid_FOV.N_points );
                end
            end

            % ensure cell array for indices_grids
            if ~iscell( indices_grids )
                indices_grids = { indices_grids };
            end

            % multiple spatial_grids_symmetric / single indices_element
            if ~isscalar( spatial_grids_symmetric ) && isscalar( indices_element )
                indices_element = repmat( indices_element, size( spatial_grids_symmetric ) );
            end

            % multiple spatial_grids_symmetric / single indices_grids
            if ~isscalar( spatial_grids_symmetric ) && isscalar( indices_grids )
                indices_grids = repmat( indices_grids, size( spatial_grids_symmetric ) );
            end

            % ensure equal number of dimensions and sizes
            auxiliary.mustBeEqualSize( spatial_grids_symmetric, indices_element, indices_grids );

            %--------------------------------------------------------------
            % 2.) shift grid positions on symmetric regular grids
            %--------------------------------------------------------------
            % specify cell array for indices_grids_shift
            indices_grids_shift = cell( size( spatial_grids_symmetric ) );

            % iterate symmetric spatial discretizations
            for index_grid = 1:numel( spatial_grids_symmetric )

                % ensure positive integers for indices_element{ index_grid }
                mustBeInteger( indices_element{ index_grid } );
                mustBePositive( indices_element{ index_grid } );

                % ensure that indices_element{ index_grid } do not exceed number of elements
                if any( indices_element{ index_grid } > numel( spatial_grids_symmetric( index_grid ).grids_elements ) )
                    errorStruct.message = sprintf( 'indices_element{ %d } exceeds number of elements!', index_grid );
                    errorStruct.identifier = 'shift_lateral:InvalidIndices';
                    error( errorStruct );
                end

                % ensure positive integers for indices_grids{ index_grid }
% TODO: check in inverse index transform
                mustBeInteger( indices_grids{ index_grid } );
                mustBePositive( indices_grids{ index_grid } );

                % ensure that indices_element{ index_grid } do not exceed number of elements
                if any( indices_grids{ index_grid } > spatial_grids_symmetric( index_grid ).grid_FOV.N_points )
                    errorStruct.message = sprintf( 'indices_grids{ %d } exceeds number of grid points!', index_grid );
                    errorStruct.identifier = 'shift_lateral:InvalidIndices';
                    error( errorStruct );
                end

                % number of dimensions (total and lateral)
                N_dimensions = spatial_grids_symmetric( index_grid ).grid_FOV.N_dimensions;
                N_dimensions_lateral = N_dimensions - 1;

                % numbers of elements along each lateral coordinate axis
                N_elements_axis = size( spatial_grids_symmetric( index_grid ).grids_elements );
                N_elements_axis = N_elements_axis( 1:N_dimensions_lateral );

                % shift in grid points required for current array element
                temp = cell( 1, N_dimensions_lateral );
                [ temp{ : } ] = ind2sub( N_elements_axis, indices_element{ index_grid }( : ) );
                indices_element_axis = cat( 2, temp{ : } );
                N_points_shift_axis = ( indices_element_axis - 1 ) .* spatial_grids_symmetric( index_grid ).N_points_per_pitch_axis;

                % subscripts of indices_grids{ index_grid }
                indices_grids_axis = inverse_index_transform( spatial_grids_symmetric( index_grid ).grid_FOV, indices_grids{ index_grid } );

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
                    index_offset_axis = spatial_grids_symmetric( index_grid ).grid_FOV.N_points_axis( 1:N_dimensions_lateral ) - ( N_elements_axis - 1 ) .* spatial_grids_symmetric( index_grid ).N_points_per_pitch_axis + 1;
                    index_offset_axis = repmat( reshape( [ index_offset_axis, 0 ], [ 1, 1, N_dimensions ] ), [ N_points_sel, N_elements_sel ] );

                    % mirror missing values
                    indices_grids_shift_axis( indicator ) = index_offset_axis( indicator ) - indices_grids_shift_axis( indicator );

                end % if any( indicator( : ) )

                % convert subscripts to linear indices
                indices_grids_shift_axis = reshape( indices_grids_shift_axis, [ N_points_sel * N_elements_sel, N_dimensions ] );
                indices_grids_shift{ index_grid } = reshape( forward_index_transform( spatial_grids_symmetric( index_grid ).grid_FOV, indices_grids_shift_axis ), [ N_points_sel, N_elements_sel ] );

            end % for index_grid = 1:numel( spatial_grids_symmetric )

            % avoid cell array for single spatial_grids_symmetric
            if isscalar( spatial_grids_symmetric )
                indices_grids_shift = indices_grids_shift{ 1 };
            end

        end % function indices_grids_shift = shift_lateral( spatial_grids_symmetric, indices_element, varargin )

	end % methods

end % classdef spatial_grid_symmetric < discretizations.spatial_grid
