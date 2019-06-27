%
% superclass for all spatial discretizations based on grids
%
% author: Martin F. Schiffner
% date: 2019-02-17
% modified: 2019-06-27
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
        % compute prefactors
        %------------------------------------------------------------------
        function prefactors = compute_prefactors( spatial_grids, axes_f )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % ensure class discretizations.spatial_grid
            if ~isa( spatial_grids, 'discretizations.spatial_grid' )
                errorStruct.message = 'spatial_grids must be discretizations.spatial_grid!';
                errorStruct.identifier = 'compute_prefactors:NoSpatialGrids';
                error( errorStruct );
            end

            % ensure equal subclasses of math.grid_regular
            auxiliary.mustBeEqualSubclasses( 'math.grid_regular', spatial_grids.grid_FOV );

            % method compute_wavenumbers ensures class math.sequence_increasing

            % multiple spatial_grids / single axes_f
            if ~isscalar( spatial_grids ) && isscalar( axes_f )
                axes_f = repmat( axes_f, size( spatial_grids ) );
            end

            % single spatial_grids / multiple axes_f
            if isscalar( spatial_grids ) && ~isscalar( axes_f )
                spatial_grids = repmat( spatial_grids, size( axes_f ) );
            end

            % ensure equal number of dimensions and sizes
            auxiliary.mustBeEqualSize( spatial_grids, axes_f );

            %--------------------------------------------------------------
            % 2.) compute prefactors
            %--------------------------------------------------------------
            % specify cell array for samples
            samples = cell( size( spatial_grids ) );

            % iterate spatial discretizations based on grids
            for index_object = 1:numel( spatial_grids )

                % geometric volume element
                delta_V = spatial_grids( index_object ).grid_FOV.cell_ref.volume;

                % compute axis of complex-valued wavenumbers
                axis_k_tilde = compute_wavenumbers( spatial_grids( index_object ).homogeneous_fluid.absorption_model, axes_f( index_object ) );

                % compute samples of prefactors
                samples{ index_object } = - delta_V * axis_k_tilde.members.^2;

            end % for index_object = 1:numel( spatial_grids )

            % create signal matrices
            prefactors = discretizations.signal_matrix( axes_f, samples );

        end % function prefactors = compute_prefactors( spatial_grids, axes_f )

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
