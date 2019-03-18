%
% superclass for all spatial discretizations based on symmetric regular grids
%
% author: Martin F. Schiffner
% date: 2019-02-17
% modified: 2019-03-18
%
classdef spatial_grid_symmetric < discretizations.spatial_grid

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % methods
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	methods

        %------------------------------------------------------------------
        % constructor
        %------------------------------------------------------------------
        function objects = spatial_grid_symmetric( grids_elements, grid_FOV )

            %--------------------------------------------------------------
            % 1.) constructor of superclass
            %--------------------------------------------------------------
            objects@discretizations.spatial_grid( grids_elements, grid_FOV );

            %--------------------------------------------------------------
            % 2.) check symmetry of FOV about the axial axis
            %--------------------------------------------------------------
            FOV_pos_ctr = 2 * grid_FOV.offset_axis( 1:(end - 1) ) + ( grid_FOV.N_points_axis( 1:(end - 1) ) - 1 ) .* grid_FOV.delta_axis( 1:(end - 1) );
            if ~all( abs( FOV_pos_ctr ) < eps )
                errorStruct.message     = 'Symmetric spatial grid requires the symmetry of FOV about the axial axis!';
                errorStruct.identifier	= 'spatial_grid_symmetric:NoSymmetry';
                error( errorStruct );
            end
            % assertion: FOV_pos_ctr is zero

            %--------------------------------------------------------------
            % 3.) lateral spacing is an integer fraction of the element pitch
            %     => translational invariance by shifts of factor_interp_tx points
            %--------------------------------------------------------------
            must_equal_integer = ( grids_elements( 2 ).positions( 1 ) - grids_elements( 1 ).positions( 1 ) ) ./ grid_FOV.delta_axis( 1:(end - 1) );
            % TODO: why is the error so large?
            if ~all( abs( must_equal_integer - floor( must_equal_integer ) ) < floor( must_equal_integer ) / 10^12 )
                errorStruct.message     = 'Symmetric spatial grid requires the lateral spacings of the grid points in the FOV to be an integer fraction of the element pitch!';
                errorStruct.identifier	= 'spatial_grid_symmetric:NoIntegerFraction';
                error( errorStruct );
            end

        end % function objects = spatial_grid_symmetric( grids_elements, grid_FOV )

	end % methods

end % classdef spatial_grid_symmetric < discretizations.spatial_grid
