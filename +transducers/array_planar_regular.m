%
% superclass for all regular planar transducer arrays
%
% author: Martin F. Schiffner
% date: 2019-08-16
% modified: 2019-08-21
%
classdef array_planar_regular < transducers.array_planar

	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	%% properties
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	properties (SetAccess = private)

        % independent properties
        face_ref ( 1, 1 ) transducers.face_planar	% elementary face
        cell_ref ( 1, 1 ) math.parallelotope        % elementary cell
        N_elements_axis ( :, 1 ) double             % number of elements along each coordinate axis (1)

	end % properties

	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	%% methods
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	methods

        %------------------------------------------------------------------
        % constructor
        %------------------------------------------------------------------
        function objects = array_planar_regular( faces_ref, cells_ref, N_elements_axis )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % ensure class transducers.face_planar
            if ~isa( faces_ref, 'transducers.face_planar' )
                errorStruct.message = 'faces_ref must be transducers.face_planar!';
                errorStruct.identifier = 'array_planar_regular:NoContinuousPlanarFaces';
                error( errorStruct );
            end

            % ensure class math.parallelotope
            if ~isa( cells_ref, 'math.parallelotope' )
                errorStruct.message = 'cells_ref must be math.parallelotope!';
                errorStruct.identifier = 'array_planar_regular:NoParallelotopes';
                error( errorStruct );
            end

            % ensure cell array for N_elements_axis
            if ~iscell( N_elements_axis )
                N_elements_axis = { N_elements_axis };
            end

            % ensure equal number of dimensions and sizes
            auxiliary.mustBeEqualSize( faces_ref, cells_ref, N_elements_axis );

            %--------------------------------------------------------------
            % 2.) create continuous regular planar transducer arrays
            %--------------------------------------------------------------
            % compute offsets
            offset_axis = cell( size( faces_ref ) );
            for index_object = 1:numel( faces_ref )
                offset_axis{ index_object } = ( 1 - N_elements_axis{ index_object } ) * ( cells_ref( index_object ).edge_lengths .* cells_ref( index_object ).basis ) / 2;
            end

            % create regular grids of center coordinates
            grids_ctr = math.grid_regular( offset_axis, cells_ref, N_elements_axis );

            % replicate planar vibrating faces to create apertures
            apertures = replicate( faces_ref, grids_ctr );

            % constructor of superclass
            objects@transducers.array_planar( apertures );

            % iterate continuous regular planar transducer arrays
            for index_object = 1:numel( objects )

                % set independent properties
                objects( index_object ).face_ref = faces_ref( index_object );
                objects( index_object ).cell_ref = cells_ref( index_object );
                objects( index_object ).N_elements_axis = N_elements_axis{ index_object };

            end % for index_object = 1:numel( objects )

        end % function objects = array_planar_regular( faces_ref, cells_ref, N_elements_axis )

        %------------------------------------------------------------------
        % forward index transform
        %------------------------------------------------------------------
        function indices_linear = forward_index_transform( arrays_planar_regular, indices_axis )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % ensure class transducers.array_planar_regular
            if ~isa( arrays_planar_regular, 'transducers.array_planar_regular' )
                errorStruct.message = 'arrays_planar_regular must be transducers.array_planar_regular!';
                errorStruct.identifier = 'forward_index_transform:NoRegularPlanarArrays';
                error( errorStruct );
            end

            % ensure cell array for indices_linear
            if ~iscell( indices_axis )
                indices_axis = { indices_axis };
            end

            % multiple arrays_planar_regular / single indices_linear
            if ~isscalar( arrays_planar_regular ) && isscalar( indices_axis )
                indices_axis = repmat( indices_axis, size( arrays_planar_regular ) );
            end

            % single arrays_planar_regular / multiple indices_linear
            if isscalar( arrays_planar_regular ) && ~isscalar( indices_axis )
                arrays_planar_regular = repmat( arrays_planar_regular, size( indices_axis ) );
            end

            % ensure equal number of dimensions and sizes
            auxiliary.mustBeEqualSize( arrays_planar_regular, indices_axis );

            %--------------------------------------------------------------
            % 2.) convert subscripts into linear indices
            %--------------------------------------------------------------
            % specify cell array for indices_linear
            indices_linear = cell( size( arrays_planar_regular ) );

            % iterate planar transducer arrays
            for index_array = 1:numel( arrays_planar_regular )

                % convert subscripts into linear indices
                temp = mat2cell( indices_axis{ index_object }, size( indices_axis{ index_object }, 1 ), ones( 1, arrays_planar_regular( index_array ).N_dimensions ) );
                indices_linear{ index_object } = sub2ind( grids_regular( index_object ).N_points_axis, temp{ : } );

            end % for index_array = 1:numel( arrays_planar_regular )

            % avoid cell array for single arrays_planar_regular
            if isscalar( arrays_planar_regular )
                indices_linear = indices_linear{ 1 };
            end

        end % function indices_linear = forward_index_transform( arrays_planar_regular, indices_axis )

        %------------------------------------------------------------------
        % inverse index transform
        %------------------------------------------------------------------
        function indices_axis = inverse_index_transform( arrays_planar_regular, indices_linear )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % ensure class transducers.array_planar_regular
            if ~isa( arrays_planar_regular, 'transducers.array_planar_regular' )
                errorStruct.message = 'arrays_planar_regular must be transducers.array_planar_regular!';
                errorStruct.identifier = 'inverse_index_transform:NoRegularPlanarArrays';
                error( errorStruct );
            end

            % ensure cell array for indices_linear
            if ~iscell( indices_linear )
                indices_linear = { indices_linear };
            end

            % multiple arrays_planar_regular / single indices_linear
            if ~isscalar( arrays_planar_regular ) && isscalar( indices_linear )
                indices_linear = repmat( indices_linear, size( arrays_planar_regular ) );
            end

            % single arrays_planar_regular / multiple indices_linear
            if isscalar( arrays_planar_regular ) && ~isscalar( indices_linear )
                arrays_planar_regular = repmat( arrays_planar_regular, size( indices_linear ) );
            end

            % ensure equal number of dimensions and sizes
            auxiliary.mustBeEqualSize( arrays_planar_regular, indices_linear );

            %--------------------------------------------------------------
            % 2.) convert linear indices into subscripts
            %--------------------------------------------------------------
            % specify cell array for indices_axis
            indices_axis = cell( size( arrays_planar_regular ) );

            % iterate planar transducer arrays
            for index_array = 1:numel( arrays_planar_regular )

                % convert linear indices into subscripts
                temp = cell( 1, arrays_planar_regular( index_array ).N_dimensions );
                [ temp{ : } ] = ind2sub( arrays_planar_regular( index_array ).N_elements_axis, indices_linear{ index_array }( : ) );
                indices_axis{ index_array } = cat( 2, temp{ : } );

            end % for index_array = 1:numel( arrays_planar_regular )

            % avoid cell array for single arrays_planar_regular
            if isscalar( arrays_planar_regular )
                indices_axis = indices_axis{ 1 };
            end

        end % function indices_axis = inverse_index_transform( arrays_planar_regular, indices_linear )

	end % methods

end % classdef array_planar_regular < transducers.array_planar
