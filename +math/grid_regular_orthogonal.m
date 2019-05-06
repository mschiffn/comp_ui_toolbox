%
% superclass for all orthogonal regular grids
%
% author: Martin F. Schiffner
% date: 2018-01-23
% modified: 2019-04-01
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

    end % methods

end % classdef grid_regular_orthogonal < math.grid_regular
