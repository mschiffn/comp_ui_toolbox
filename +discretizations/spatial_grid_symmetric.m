%
% superclass for all symmetric spatial discretizations based on orthogonal regular grids
%
% author: Martin F. Schiffner
% date: 2019-02-17
% modified: 2019-04-08
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
                if ~isa( grids_elements{ index_object }, 'discretizations.grid_regular_orthogonal' )
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

                %----------------------------------------------------------
                % a) check lateral symmetry of FOV about the axial axis
                %----------------------------------------------------------
                FOV_pos_ctr = 2 * grids_FOV( index_object ).offset_axis( 1:(end - 1) ) + ( grids_FOV( index_object ).N_points_axis( 1:(end - 1) ) - 1 ) .* grids_FOV( index_object ).cell_ref.edge_lengths( 1:(end - 1) );
                if ~all( abs( double( FOV_pos_ctr ) ) < eps )
                    errorStruct.message     = 'Symmetric spatial grid requires the symmetry of FOV about the axial axis!';
                    errorStruct.identifier	= 'spatial_grid_symmetric:NoSymmetry';
                    error( errorStruct );
                end

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

	end % methods

end % classdef spatial_grid_symmetric < discretizations.spatial_grid
