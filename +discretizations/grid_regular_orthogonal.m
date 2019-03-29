%
% superclass for all regular grids using cuboids
%
% author: Martin F. Schiffner
% date: 2018-01-23
% modified: 2019-03-23
%
classdef grid_regular_orthogonal < discretizations.grid_regular

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % methods
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	methods

        %------------------------------------------------------------------
        % constructor
        %------------------------------------------------------------------
        function objects = grid_regular_orthogonal( offset_axis, delta_axis, N_points_axis )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % reference cells are cuboids
            cells_ref = math.parallelotope( delta_axis );

            %--------------------------------------------------------------
            % 2.) constructor of superclass
            %--------------------------------------------------------------
            objects@discretizations.grid_regular( offset_axis, cells_ref, N_points_axis );

        end % function objects = grid_regular_orthogonal( offset_axis, delta_axis, N_points_axis )

    end % methods

end % classdef grid_regular_orthogonal < discretizations.grid_regular
