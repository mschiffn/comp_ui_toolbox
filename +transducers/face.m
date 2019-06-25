%
% superclass for all vibrating faces
%
% author: Martin F. Schiffner
% date: 2019-02-18
% modified: 2019-06-04
%
classdef face

	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	%% properties
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	properties ( SetAccess = private )

        % independent properties
        shape % ( 1, 1 ) math.shape
        apodization ( 1, 1 ) function_handle = @( pos_rel_norm ) ones( size( pos_rel_norm, 1 ), 1 );	% apodization along each coordinate axis (pos_rel_norm = normalized relative positions of the grid points)
        lens ( 1, 1 ) transducers.lens

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

            % property validation function ensures class math.shape for shapes

            % ensure cell array for apodizations
            if ~iscell( apodizations )
                apodizations = { apodizations };
            end

            % property validation function ensures class transducers.lens for lenses

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

            % specify cell array for positions_ctr
            positions_ctr = cell( numel( faces ), 1 );

            % iterate planar vibrating faces
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
        % replicate faces
        %------------------------------------------------------------------
        function out = replicate( faces, grids )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % ensure class transducers.face
            if ~isa( faces, 'transducers.face' )
                errorStruct.message = 'faces must be transducers.face!';
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
            auxiliary.mustBeEqualSize( faces, grids );

            %--------------------------------------------------------------
            % 2.) check arguments
            %--------------------------------------------------------------
            % specify cell array for out
            out = cell( size( faces ) );

            % iterate faces
            for index_object = 1:numel( faces )

                % repeat current face
                out{ index_object } = repmat( faces( index_object ), [ grids( index_object ).N_points, 1 ] );

                % move shapes into specified positions
                shapes = move( [ out{ index_object }.shape ]', mat2cell( grids.positions, ones( grids( index_object ).N_points, 1 ), grids( index_object ).N_dimensions ) );

                % assign shapes
                for index = 1:grids( index_object ).N_points
                    out{ index_object }( index ).shape = shapes( index );
                end

            end % for index_object = 1:numel( faces )

        end % function out = replicate( faces, grids )

        %------------------------------------------------------------------
        % discretize
        %------------------------------------------------------------------
        function structs_out = discretize( faces, parameters )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % ensure class transducers.face
            if ~isa( faces, 'transducers.face' )
                errorStruct.message = 'faces must be transducers.face!';
                errorStruct.identifier = 'discretize:NoFaces';
                error( errorStruct );
            end

            % method discretize in shape ensures class discretizations.parameters for options_elements

            %--------------------------------------------------------------
            % 2.) discretize vibrating faces
            %--------------------------------------------------------------
            % discretize shapes
            grids_act = discretize( reshape( [ faces.shape ], size( faces ) ), parameters );

            % specify cell arrays for apodization and time delays
            apodization_act = cell( size( grids_act ) );
            time_delays_act = cell( size( grids_act ) );

            % iterate vibrating faces
            for index_element = 1:numel( grids_act )

                % compute normalized relative positions of the grid points
                positions_rel = grids_act( index_element ).positions - center( faces( index_element ).shape );
                positions_rel_norm = 2 * positions_rel ./ abs( faces( index_element ).shape.intervals );

                % compute apodization weights
                apodization_act{ index_element } = faces( index_element ).apodization( positions_rel_norm );

                % compute time delays for each coordinate axis
% TODO: compute in spatial transfer function
                time_delays_act{ index_element } = faces( index_element ).lens.thickness( positions_rel_norm ) / faces( index_element ).lens.absorption_model.c_0;

            end % for index_element = 1:numel( grids_act )

            % create structures
            structs_out = struct( 'grid', num2cell( grids_act ), 'apodization', apodization_act, 'time_delays', time_delays_act );

        end % function structs_out = discretize( faces, parameters )

    end % methods

end % classdef face
