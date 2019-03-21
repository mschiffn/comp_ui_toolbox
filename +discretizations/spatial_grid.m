%
% superclass for all spatial discretizations based on regular grids
%
% author: Martin F. Schiffner
% date: 2019-02-17
% modified: 2019-03-20
%
classdef spatial_grid < discretizations.spatial

	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	% properties
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	properties (SetAccess = private)

        % independent properties
        grids_elements ( 1, : ) discretizations.grid	% regular grids representing the array elements
        grid_FOV ( 1, 1 ) discretizations.grid          % regular grid representing the field-of-view

	end % properties

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % methods
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	methods

        %------------------------------------------------------------------
        % constructor
        %------------------------------------------------------------------
        function objects = spatial_grid( grids_elements, grid_FOV )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % ensure cell array
            if ~iscell( grids_elements )
                grids_elements = { grids_elements };
            end

            % ensure class discretizations.grid
            if ~isa( grid_FOV, 'discretizations.grid' )
                errorStruct.message     = 'grid_FOV must be discretizations.grid!';
                errorStruct.identifier	= 'spatial_grid:NoGrid';
                error( errorStruct );
            end
            % assertion: grid_FOV is discretizations.grid

            % ensure equal number of dimensions and sizes
            auxiliary.mustBeEqualSize( grids_elements, grid_FOV );
            % assertion: grids_elements and grid_FOV have the same size

            %--------------------------------------------------------------
            % 2.) constructor of superclass
            %--------------------------------------------------------------
            objects@discretizations.spatial();
            objects = repmat( objects, size( grids_elements ) );

            %--------------------------------------------------------------
            % 3.) compute mutual distances for array elements
            %--------------------------------------------------------------
            for index_object = 1:numel( objects )

                % ensure class discretizations.grid
                if ~isa( grids_elements{ index_object }, 'discretizations.grid' )
                    errorStruct.message     = sprintf( 'grids_elements{ %d } must be discretizations.grid!', index_object );
                    errorStruct.identifier	= 'spatial_grid:NoGrid';
                    error( errorStruct );
                end
                % assertion: grids_elements{ index_object } is discretizations.grid

                % set independent properties
                objects( index_object ).grids_elements = grids_elements{ index_object };
                objects( index_object ).grid_FOV = grid_FOV( index_object );

            end % for index_object = 1:numel( objects )

        end % function objects = spatial_grid( grids_elements, grid_FOV )

	end % methods

end % classdef spatial_grid < discretizations.spatial
