%
% superclass for all spatial discretizations based on grids
%
% author: Martin F. Schiffner
% date: 2019-02-17
% modified: 2019-06-06
%
classdef spatial_grid < discretizations.spatial

	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	%% properties
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	properties (SetAccess = private)

        % independent properties
        grids_elements ( :, 1 ) %math.grid	% grids representing the array elements, apodization weights, and focal distances
        grid_FOV ( 1, 1 ) math.grid         % grid representing the field-of-view

	end % properties

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% methods
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	methods

        %------------------------------------------------------------------
        % constructor
        %------------------------------------------------------------------
        function objects = spatial_grid( homogeneous_fluids, strs_name, grids_elements, grids_FOV )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % superclass ensures class pulse_echo_measurements.homogeneous_fluid

            % ensure cell array for grids_elements
            if ~iscell( grids_elements )
                grids_elements = { grids_elements };
            end

            % ensure class math.grid
            if ~isa( grids_FOV, 'math.grid' )
                errorStruct.message     = 'grids_FOV must be math.grid!';
                errorStruct.identifier	= 'spatial_grid:NoGrid';
                error( errorStruct );
            end

            % ensure equal number of dimensions and sizes
            auxiliary.mustBeEqualSize( homogeneous_fluids, grids_elements, grids_FOV );

            %--------------------------------------------------------------
            % 2.) constructor of superclass
            %--------------------------------------------------------------
            objects@discretizations.spatial( homogeneous_fluids, strs_name );

            %--------------------------------------------------------------
            % 3.) check and set independent properties
            %--------------------------------------------------------------
            for index_object = 1:numel( objects )

                % ensure class math.grid
% TODO: introduce class for discretized face
% TODO: validate N_dimensions, i.e. difference of unity!
                if ~( isa( [ grids_elements{ index_object }.grid ], 'math.grid' ) && isa( [ grids_elements{ index_object }.time_delays ], 'physical_values.time' ) )
                    errorStruct.message     = sprintf( 'grids_elements{ %d } must be math.grid!', index_object );
                    errorStruct.identifier	= 'spatial_grid:NoGrid';
                    error( errorStruct );
                end

                % set independent properties
                objects( index_object ).grids_elements = grids_elements{ index_object };
                objects( index_object ).grid_FOV = grids_FOV( index_object );

            end % for index_object = 1:numel( objects )

        end % function objects = spatial_grid( homogeneous_fluids, strs_name, grids_elements, grids_FOV )
      
        %------------------------------------------------------------------
        % check for symmetry
        %------------------------------------------------------------------
        function tf = issymmetric( spatial_grids )

            % initialize results with false
            tf = false( size( spatial_grids ) );

            % iterate spatial discretizations based on grids
            for index_object = 1:numel( spatial_grids )

                % TODO: check for symmetry
            end

        end % function tf = issymmetric( spatial_grids )

	end % methods

end % classdef spatial_grid < discretizations.spatial
