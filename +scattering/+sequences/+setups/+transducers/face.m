%
% superclass for all vibrating faces
%
% author: Martin F. Schiffner
% date: 2019-02-18
% modified: 2019-10-25
%
classdef face

	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	%% properties
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	properties (SetAccess = private)

        % independent properties
        shape ( 1, 1 ) scattering.sequences.setups.geometry.shape { mustBeNonempty } = scattering.sequences.setups.geometry.orthotope % shape of the vibrating face
        apodization ( :, 1 ) = @scattering.sequences.setups.transducers.apodization.uniform % apodization along each coordinate axis
        lens ( 1, 1 ) scattering.sequences.setups.transducers.lens % acoustic lens
% TODO: matching layers
	end % properties

	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	%% methods
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	methods

        %------------------------------------------------------------------
        % constructor
        %------------------------------------------------------------------
        function objects = face( shapes, apodizations, lenses )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % return for no input arguments
            if nargin == 0
                return;
            end

            % ensure class scattering.sequences.setups.geometry.shape
            if ~isa( shapes, 'scattering.sequences.setups.geometry.shape' )
                errorStruct.message = 'shapes must be scattering.sequences.setups.geometry.shape!';
                errorStruct.identifier = 'face:NoShapes';
                error( errorStruct );
            end

            % ensure cell array for apodizations
            if ~iscell( apodizations )
                apodizations = { apodizations };
            end

% TODO: lens
            % property validation function ensures class scattering.sequences.setups.transducers.lens for lenses

            % ensure equal number of dimensions and sizes
            auxiliary.mustBeEqualSize( shapes, apodizations, lenses );

            %--------------------------------------------------------------
            % 2.) create vibrating faces
            %--------------------------------------------------------------
            % repeat default vibrating face
            objects = repmat( objects, size( shapes ) );

            % iterate vibrating faces
            for index_object = 1:numel( objects )

                % ensure class function_handle
                if ~isa( apodizations{ index_object }, 'function_handle' )
                    errorStruct.message = sprintf( 'apodizations{ %d } must be function_handle!', index_object );
                    errorStruct.identifier = 'face:NoFunctionHandle';
                    error( errorStruct );
                end

                % set independent properties
                objects( index_object ).shape = shapes( index_object );
                objects( index_object ).apodization = apodizations{ index_object };
                objects( index_object ).lens = lenses( index_object );

            end % for index_object = 1:numel( objects )

        end % function objects = face( shapes, apodizations, lenses )

        %------------------------------------------------------------------
        % numbers of dimensions
        %------------------------------------------------------------------
        function results = get_N_dimensions( faces )

            % extract numbers of dimensions
            results = cellfun( @( x ) x.N_dimensions, { faces.shape } );

        end % function results = N_dimensions( faces )

        %------------------------------------------------------------------
        % center
        %------------------------------------------------------------------
        function positions_ctr = center( faces )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % ensure class scattering.sequences.setups.transducers.face
            if ~isa( faces, 'scattering.sequences.setups.transducers.face' )
                errorStruct.message = 'faces must be scattering.sequences.setups.transducers.face!';
                errorStruct.identifier = 'center:NoFaces';
                error( errorStruct );
            end

            %--------------------------------------------------------------
            % 2.) compute centers of vibrating faces
            %--------------------------------------------------------------
            % specify cell array for positions_ctr
            positions_ctr = cell( numel( faces ), 1 );

            % iterate vibrating faces
            for index_object = 1:numel( faces )
                positions_ctr{ index_object } = center( faces( index_object ).shape );
            end

            % check numbers of dimensions
            N_dimensions = cellfun( @numel, positions_ctr );
            if all( N_dimensions( : ) == N_dimensions( 1 ) )
                positions_ctr = cat( 1, positions_ctr{ : } );
            end

        end % function positions_ctr = center( faces )

        %------------------------------------------------------------------
        % replicate reference faces
        %------------------------------------------------------------------
        function apertures = replicate( faces_ref, grids )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % ensure class scattering.sequences.setups.transducers.face
            if ~isa( faces_ref, 'scattering.sequences.setups.transducers.face' )
                errorStruct.message = 'faces_ref must be scattering.sequences.setups.transducers.face!';
                errorStruct.identifier = 'replicate:NoFaces';
                error( errorStruct );
            end

            % ensure class math.grid
            if ~isa( grids, 'math.grid' )
                errorStruct.message = 'grids must be math.grid!';
                errorStruct.identifier = 'replicate:NoGrids';
                error( errorStruct );
            end

            % ensure equal number of dimensions and sizes
            auxiliary.mustBeEqualSize( faces_ref, grids );

            %--------------------------------------------------------------
            % 2.) replicate reference faces
            %--------------------------------------------------------------
            % specify cell array for apertures
            apertures = cell( size( faces_ref ) );

            % iterate vibrating reference faces
            for index_object = 1:numel( faces_ref )

                % repeat current reference face
                apertures{ index_object } = repmat( faces_ref( index_object ), [ grids( index_object ).N_points, 1 ] );

                % move shapes into specified positions
                shapes = move( [ apertures{ index_object }.shape ]', mat2cell( grids.positions, ones( grids( index_object ).N_points, 1 ), grids( index_object ).N_dimensions ) );

                % assign moved shapes
                for index = 1:grids( index_object ).N_points
                    apertures{ index_object }( index ).shape = shapes( index );
                end

            end % for index_object = 1:numel( faces_ref )

            % avoid cell array for single faces_ref
            if isscalar( faces_ref )
                apertures = apertures{ 1 };
            end

        end % function apertures = replicate( faces_ref, grids )

        %------------------------------------------------------------------
        % discretize
        %------------------------------------------------------------------
        function faces = discretize( faces, methods )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % ensure class scattering.sequences.setups.transducers.face
            if ~isa( faces, 'scattering.sequences.setups.transducers.face' )
                errorStruct.message = 'faces must be scattering.sequences.setups.transducers.face!';
                errorStruct.identifier = 'discretize:NoFaces';
                error( errorStruct );
            end

            % method discretize in shape ensures shape and class scattering.sequences.setups.discretizations.methods.method for methods

            %--------------------------------------------------------------
            % 2.) discretize vibrating faces
            %--------------------------------------------------------------
            % discretize shapes
%             cellfun( @( x ) discretize( x, methods ), { faces.shape } )
            shapes_discrete = discretize( reshape( [ faces.shape ], size( faces ) ), methods );

            % compute centers of vibrating faces
            positions_ctr = center( faces );

            % iterate vibrating faces
            for index_face = 1:numel( shapes_discrete )

                % assign discrete shape
                faces( index_face ).shape = shapes_discrete( index_face );

                % compute normalized relative positions of the grid points
% TODO: only valid for grids!
% TODO: only valid for orthotope!
% TODO: move to method in discretized shape?
                positions_rel = faces( index_face ).shape.grid.positions - positions_ctr( index_face, : );
                positions_rel_norm = 2 * positions_rel ./ abs( faces( index_face ).shape.intervals );

                % evaluate apodization weights
                faces( index_face ).apodization = faces( index_face ).apodization( positions_rel_norm );

                % discretize acoustic lens
                faces( index_face ).lens = discretize( faces( index_face ).lens, abs( faces( index_face ).shape.intervals ), positions_rel_norm );

            end % for index_face = 1:numel( shapes_discrete )

        end % function faces = discretize( faces, methods )

        %------------------------------------------------------------------
        % compute complex-valued apodization weights
        %------------------------------------------------------------------
        function weights = compute_weights( faces, axes_f )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % ensure class scattering.sequences.setups.transducers.face
            if ~isa( faces, 'scattering.sequences.setups.transducers.face' )
                errorStruct.message = 'faces must be scattering.sequences.setups.transducers.face!';
                errorStruct.identifier = 'compute_weights:NoFaces';
                error( errorStruct );
            end

            % ensure class math.sequence_increasing with physical_values.frequency members
            if ~( isa( axes_f, 'math.sequence_increasing' ) && all( cellfun( @( x ) isa( x, 'physical_values.frequency' ), { axes_f.members } ) ) )
                errorStruct.message = 'axes_f must be math.sequence_increasing with physical_values.frequency members!';
                errorStruct.identifier = 'compute_weights:InvalidFrequencyAxes';
                error( errorStruct );
            end

            % multiple faces / single axes_f
            if ~isscalar( faces ) && isscalar( axes_f )
                axes_f = repmat( axes_f, size( faces ) );
            end

            % single faces / multiple axes_f
            if isscalar( faces ) && ~isscalar( axes_f )
                faces = repmat( faces, size( axes_f ) );
            end

            % ensure equal number of dimensions and sizes
            auxiliary.mustBeEqualSize( faces, axes_f );

            %--------------------------------------------------------------
            % 2.) compute complex-valued apodization weights
            %--------------------------------------------------------------
            % specify cell array for weights
            weights = cell( size( faces ) );

            % iterate discretized vibrating faces
            for index_face = 1:numel( faces )

                % compute complex-valued apodization weights (scalar)
                weights{ index_face } = compute_weights_scalar( faces( index_face ), axes_f( index_face ) );

            end % for index_face = 1:numel( faces )

            % avoid cell array for single faces
            if isscalar( faces )
                weights = weights{ 1 };
            end

        end % function weights = compute_weights( faces, axes_f )

        %------------------------------------------------------------------
        % is symmetric
        %------------------------------------------------------------------
        function tf = issymmetric( faces )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % ensure class scattering.sequences.setups.transducers.face
            if ~isa( faces, 'scattering.sequences.setups.transducers.face' )
                errorStruct.message = 'faces must be scattering.sequences.setups.transducers.face!';
                errorStruct.identifier = 'issymmetric:NoFaces';
                error( errorStruct );
            end

            % ensure class scattering.sequences.setups.geometry.orthotope_grid
            indicator = cellfun( @( x ) ~isa( x, 'scattering.sequences.setups.geometry.orthotope_grid' ), { faces.shape } );
            if any( indicator( : ) )
                errorStruct.message = 'faces.shape must be scattering.sequences.setups.geometry.orthotope_grid!';
                errorStruct.identifier = 'issymmetric:NoGridOrthotopes';
                error( errorStruct );
            end

            % ensure class math.grid_regular_orthogonal
            indicator = cellfun( @( x ) ~isa( x.grid, 'math.grid_regular_orthogonal' ), { faces.shape } );
            if any( indicator( : ) )
                errorStruct.message = 'The grids representing faces.shape must be math.grid_regular_orthogonal!';
                errorStruct.identifier = 'issymmetric:NoOrthogonalRegularGrids';
                error( errorStruct );
            end

            %--------------------------------------------------------------
            % 2.) check arguments
            %--------------------------------------------------------------
            % initialize tf w/ true
            tf = true( size( faces ) );
% TODO: lateral symmetry of grid?
            % iterate vibrating faces
            for index_face = 1:numel( faces )

                % numbers of grid points on each side of the center
                M_points_axis = ceil( ( faces( index_face ).shape.grid.N_points_axis - 1 ) / 2 );

                % ensure lateral symmetries of the apodization
                apodization = reshape( faces( index_face ).apodization, faces( index_face ).shape.grid.N_points_axis );
                indicator_1 = ( apodization( 1:M_points_axis( 1 ), : ) - apodization( end:-1:( end - M_points_axis( 1 ) + 1 ), : ) ) > eps( 0 );
                indicator_2 = ( apodization( :, 1:M_points_axis( 2 ) ) - apodization( :, end:-1:( end - M_points_axis( 2 ) + 1 ) ) ) > eps( 0 );

                % ensure lateral symmetries of the apodization
                thickness = reshape( faces( index_face ).lens.thickness, faces( index_face ).shape.grid.N_points_axis );
                indicator_3 = double( thickness( 1:M_points_axis( 1 ), : ) - thickness( end:-1:( end - M_points_axis( 1 ) + 1 ), : ) ) > eps( 0 );
                indicator_4 = double( thickness( :, 1:M_points_axis( 2 ) ) - thickness( :, end:-1:( end - M_points_axis( 2 ) + 1 ) ) ) > eps( 0 );

                % check symmetry
                if any( indicator_1( : ) ) || any( indicator_2( : ) ) || any( indicator_3( : ) ) || any( indicator_4( : ) )
                    tf( index_face ) = false;
                end

            end % for index_face = 1:numel( faces )

        end % function tf = issymmetric( faces )

    end % methods

	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	%% methods (private and hidden)
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	methods (Access = private, Hidden)

        %------------------------------------------------------------------
        % compute complex-valued apodization weights (scalar)
        %------------------------------------------------------------------
        function weights = compute_weights_scalar( face, axis_f )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % calling function ensures class scattering.sequences.setups.transducers.face (scalar) for face
            % calling function ensures class math.sequence_increasing with physical_values.frequency members (scalar) for axis_f

            %--------------------------------------------------------------
            % 2.) compute complex-valued apodization weights (scalar)
            %--------------------------------------------------------------
            % compute complex-valued wavenumbers for acoustic lens
            axis_k_tilde_lens = compute_wavenumbers( face.lens.absorption_model, axis_f );

            % compute complex-valued apodization weights
            weights = reshape( face.apodization .* exp( - 1j * face.lens.thickness * axis_k_tilde_lens.members.' ), [ face.shape.grid.N_points, 1, abs( axis_f ) ] );

        end % function weights = compute_weights_scalar( face, axis_f )

	end % methods (Access = private, Hidden)

end % classdef face
