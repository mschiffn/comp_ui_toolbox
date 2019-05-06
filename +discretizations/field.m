%
% superclass for all fields
%
% author: Martin F. Schiffner
% date: 2019-01-22
% modified: 2019-04-30
%
classdef field < discretizations.signal_matrix

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% properties
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	properties (SetAccess = private)

        % independent properties
        grid_FOV ( 1, 1 ) discretizations.grid

        % dependent properties
        size_bytes ( 1, 1 ) physical_values.byte	% memory consumption

    end % properties

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% methods
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	methods

        %------------------------------------------------------------------
        % constructor
        %------------------------------------------------------------------
        function objects = field( axes, grids_FOV, samples )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % superclass ensures class math.sequence_increasing for axes

            % ensure class discretizations.grid
            if ~isa( grids_FOV, 'discretizations.grid' )
                errorStruct.message = 'grids_FOV must be discretizations.grid!';
                errorStruct.identifier = 'field:NoGrids';
                error( errorStruct );
            end

            % ensure equal number of dimensions and sizes
            auxiliary.mustBeEqualSize( axes, grids_FOV );

            %--------------------------------------------------------------
            % 2.) create fields
            %--------------------------------------------------------------
            % constructor of superclass
            objects@discretizations.signal_matrix( axes, samples );

            % iterate fields
            for index_object = 1:numel( objects )

                % ensure correct size of grids_FOV
                if grids_FOV( index_object ).N_points ~= objects( index_object ).N_signals
                    errorStruct.message = sprintf( 'Number of grid points in grids_FOV( %d ) must equal the number of signals %d!', index_object, objects( index_object ).N_signals );
                    errorStruct.identifier = 'field:GridSizeMismatch';
                    error( errorStruct );
                end

                % set independent properties
                objects( index_object ).grid_FOV = grids_FOV( index_object );

                % compute memory consumption
                objects( index_object ).size_bytes = data_volume( objects( index_object ) );

            end % for index_object = 1:numel( objects )

        end % function objects = field( axes, grids_FOV, samples )

        %------------------------------------------------------------------
        % lateral shift (TODO: check for correctness)
        %------------------------------------------------------------------
        function output = shift_lateral( fields, spatial_grids_symmetric, indices_element, varargin )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % ensure class discretizations.spatial_grid_symmetric
            if ~isa( spatial_grids_symmetric, 'discretizations.spatial_grid_symmetric' )
                errorStruct.message = 'spatial_grids_symmetric must be discretizations.spatial_grid_symmetric!';
                errorStruct.identifier = 'shift:NoSymmetricSpatialGrids';
                error( errorStruct );
            end

            % ensure cell array for indices_element
            if ~iscell( indices_element )
                indices_element = { indices_element };
            end

            % multiple fields / single spatial_grids_symmetric
            if ~isscalar( fields ) && isscalar( spatial_grids_symmetric )
                spatial_grids_symmetric = repmat( spatial_grids_symmetric, size( fields ) );
            end

            % multiple fields / single indices_element
            if ~isscalar( fields ) && isscalar( indices_element )
                indices_element = repmat( indices_element, size( fields ) );
            end

            %--------------------------------------------------------------
            % 2.) compute shifted indices
            %--------------------------------------------------------------
            indices_shift = shift_lateral( spatial_grids_symmetric, indices_element, varargin{ : } );

            % ensure cell array for indices_shift
            if ~iscell( indices_shift )
                indices_shift = { indices_shift };
            end

            %--------------------------------------------------------------
            % 3.) resample fields
            %--------------------------------------------------------------
            % specify cell array for output
            output = cell( size( fields ) );

            % iterate fields
            for index_field = 1:numel( fields )

                % ensure identical grids
                if ~isequal( fields( index_field ).grid_FOV, spatial_grids_symmetric( index_field ).grid_FOV )
                    errorStruct.message = sprintf( 'fields( %d ).grid_FOV differs from spatial_grids_symmetric( %d ).grid_FOV!', index_field, index_field );
                    errorStruct.identifier = 'shift:UnequalSpatialGrids';
                    error( errorStruct );
                end

                % repeat current field
                output{ index_field } = repmat( fields( index_field ), size( indices_element{ index_field } ) );

                % iterate shifts
                for index_shift = 1:numel( indices_element{ index_field } )

                    % resample according to current shift
                    output{ index_field }( index_shift ).samples = fields( index_field ).samples( indices_shift{ index_field }( :, index_shift ), : );

                end % for index_shift = 1:numel( indices_element{ index_field } )

            end % for index_field = 1:numel( fields )

            % avoid cell array for single fields
            if isscalar( fields )
                output = output{ 1 };
            end

        end % function output = shift_lateral( fields, spatial_grids_symmetric, indices_element, varargin )

        %------------------------------------------------------------------
        % subsample (overload subsample method)
        %------------------------------------------------------------------
        function fields = subsample( fields, indices_axes, indices_grids )

            %--------------------------------------------------------------
            % 1.) use subsampling method of superclass
            %--------------------------------------------------------------
% TODO: do not use superclass method for efficiency
            fields = subsample@discretizations.signal_matrix( fields, indices_axes );

            %--------------------------------------------------------------
            % 2.) check arguments
            %--------------------------------------------------------------
            % return for less than three arguments or empty indices_grids
            if nargin < 3 || isempty( indices_grids )
                return;
            end

            % ensure cell array for indices_grids
            if ~iscell( indices_grids )
                indices_grids = { indices_grids };
            end

            % ensure equal number of dimensions and sizes
            auxiliary.mustBeEqualSize( fields, indices_grids );

            %--------------------------------------------------------------
            % 3.) perform subsampling of grid
            %--------------------------------------------------------------
            % subsample grids
            grids_FOV = subgrid( [ fields.grid_FOV ], indices_grids );

            % iterate fields
            for index_object = 1:numel( fields )

                % assign subsampled grids
                fields( index_object ).grid_FOV = grids_FOV( index_object );

                % subsample samples
                fields( index_object ).samples = fields( index_object ).samples( indices_grids{ index_object }, : );
                fields( index_object ).N_signals = numel( indices_grids{ index_object } );
                fields( index_object ).size_bytes = data_volume( fields( index_object ) );

            end % for index_object = 1:numel( fields )

        end % function fields = subsample( fields, indices_axes, indices_grids )

        %------------------------------------------------------------------
        % TODO: times
        %------------------------------------------------------------------

        %------------------------------------------------------------------
        % 2-D plots (overload show function)
        %------------------------------------------------------------------
        function hdl = show( fields )

            %--------------------------------------------------------------
            % 1.) display fields
            %--------------------------------------------------------------
            % initialize hdl with zeros
            hdl = zeros( size( fields ) );

            % iterate fields
            for index_object = 1:numel( fields )

                % number of samples
                N_samples = abs( fields( index_object ).axis );
                index_shift = ceil( N_samples / 2 );

                % create figure
                hdl( index_object ) = figure( index_object );

                % ensure class discretizations.grid_regular
                if ~isa( fields( index_object ).grid_FOV, 'discretizations.grid_regular' )
                    show@discretizations.signal_matrix( fields( index_object ) );
                    continue;
                end

                % extract and reshape samples
                samples_act = reshape( fields( index_object ).samples, [ fields( index_object ).grid_FOV.N_points_axis, N_samples ] );

                % check number of spatial dimensions
                switch fields( index_object ).grid_FOV.N_dimensions

                    case 2

                        %--------------------------------------------------
                        % a) two-dimensional Euclidean space
                        %--------------------------------------------------
                        axis_1 = double( fields( index_object ).grid_FOV.positions([1,262], 1) );
                        axis_2 = double( fields( index_object ).grid_FOV.positions([1,68644], 2) );
                        subplot( 2, 3, 1);
                        imagesc( axis_2, axis_1, abs( double( samples_act( :, :, 1 ) ) ) );
                        subplot( 2, 3, 2);
                        imagesc( axis_2, axis_1, abs( double( samples_act( :, :, index_shift ) ) ) );
                        subplot( 2, 3, 3);
                        imagesc( axis_2, axis_1, abs( double( samples_act( :, :, end ) ) ) );
                        subplot( 2, 3, 4);
                        imagesc( axis_2, axis_1, angle( double( samples_act( :, :, 1 ) ) ) );
                        subplot( 2, 3, 5);
                        imagesc( axis_2, axis_1, angle( double( samples_act( :, :, index_shift ) ) ) );
                        subplot( 2, 3, 6);
                        imagesc( axis_2, axis_1, angle( double( samples_act( :, :, end ) ) ) );

                    case 3

                        %--------------------------------------------------
                        % b) three-dimensional Euclidean space
                        %--------------------------------------------------
                        index_shift_pos = ceil( fields( index_object ).grid_FOV.N_points_axis / 2 );
                        axis_1 = double( fields( index_object ).grid_FOV.positions( [1,262], 1 ) );
                        axis_2 = double( fields( index_object ).grid_FOV.positions( [1,68644], 2 ) );
                        axis_3 = double( fields( index_object ).grid_FOV.positions( [1,68644], 3 ) );
                        subplot( 3, 3, 1);
                        imagesc( abs( double( squeeze( samples_act( :, index_shift_pos( 2 ), :, 1 ) ) ) ) );
                        subplot( 3, 3, 2);
                        imagesc( abs( double( squeeze( samples_act( :, index_shift_pos( 2 ), :, index_shift ) ) ) ) );
                        subplot( 3, 3, 3);
                        imagesc( abs( double( squeeze( samples_act( :, index_shift_pos( 2 ), :, end ) ) ) ) );
                        subplot( 3, 3, 4);
                        imagesc( abs( double( squeeze( samples_act( :, :, 51, 1 ) ) ) ) );
                        subplot( 3, 3, 5);
                        imagesc( abs( double( squeeze( samples_act( :, :, 51, index_shift ) ) ) ) );
                        subplot( 3, 3, 6);
                        imagesc( abs( double( squeeze( samples_act( :, :, 51, end ) ) ) ) );
                        subplot( 3, 3, 7);
                        imagesc( abs( double( squeeze( samples_act( index_shift_pos( 1 ), :, :, 1 ) ) ) ) );
                        subplot( 3, 3, 8);
                        imagesc( abs( double( squeeze( samples_act( index_shift_pos( 1 ), :, :, index_shift ) ) ) ) );
                        subplot( 3, 3, 9);
                        imagesc( abs( double( squeeze( samples_act( index_shift_pos( 1 ), :, :, end ) ) ) ) );

                    otherwise

                        %--------------------------------------------------
                        % c) dimensionality not implemented
                        %--------------------------------------------------
                        errorStruct.message     = 'Number of dimensions not implemented!';
                        errorStruct.identifier	= 'show:UnknownDimensions';
                        error( errorStruct );

                end % switch fields( index_object ).grid_FOV.N_dimensions
                
            end % for index_object = 1:numel( fields )

        end % function hdl = show( fields )

	end % methods

end % classdef field < discretizations.signal_matrix
