%
% superclass for all planar transducer arrays
%
% author: Martin F. Schiffner
% date: 2017-04-19
% modified: 2019-04-24
%
classdef array_planar < transducers.array

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% properties
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	properties (SetAccess = private)

        % dependent properties
        element_pitch_axis ( 1, : ) physical_values.length	% pitches along each coordinate axis (m)
        aperture ( :, : ) transducers.face_planar           % aperture

	end % properties

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% methods
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	methods

        %------------------------------------------------------------------
        % constructor
        %------------------------------------------------------------------
        function objects = array_planar( parameters_planar, N_dimensions )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % ensure class transducers.parameters_planar
            if ~isa( parameters_planar, 'transducers.parameters_planar' )
                errorStruct.message     = 'parameters_planar must be transducers.parameters_planar!';
                errorStruct.identifier	= 'array_planar:NoParametersPlanar';
                error( errorStruct );
            end

            %--------------------------------------------------------------
            % 2.) constructor of superclass
            %--------------------------------------------------------------
            objects@transducers.array( parameters_planar, N_dimensions );

            %--------------------------------------------------------------
            % 3.) set dependent properties
            %--------------------------------------------------------------
            for index_object = 1:numel( objects )

                objects( index_object ).element_pitch_axis = element_pitch_axis( objects( index_object ).parameters );
                objects( index_object ).aperture = create_aperture( objects( index_object ) );

            end % for index_object = 1:numel( objects )

        end % function objects = array_planar( parameters_planar, N_dimensions )

        %------------------------------------------------------------------
        % create aperture (TODO: replicate reference face_planar)
        %------------------------------------------------------------------
        function apertures = create_aperture( arrays_planar )

            % initialize cell array
            apertures = cell( size( arrays_planar ) );

            for index_object = 1:numel( arrays_planar )

                % create position intervals for aperture
                M_elements_axis = ( arrays_planar( index_object ).parameters.N_elements_axis - 1 ) / 2;
                intervals = cell( 1, arrays_planar( index_object ).N_dimensions );

                for index_dimension = 1:arrays_planar( index_object ).N_dimensions

                    % compute lower and upper bounds on the position intervals on current axis
                    pos_ctr = ( -M_elements_axis( index_dimension ):M_elements_axis( index_dimension ) ) * arrays_planar( index_object ).element_pitch_axis( index_dimension );
                    pos_rel = arrays_planar( index_object ).parameters.element_width_axis( index_dimension ) / 2;
                    lbs = pos_ctr - pos_rel;
                    ubs = pos_ctr + pos_rel;

                    % create position intervals
                    shape_act = ones( 1, max( [ arrays_planar( index_object ).N_dimensions, 2 ] ) );
                    shape_act( index_dimension ) = arrays_planar( index_object ).parameters.N_elements_axis( index_dimension );
                    rep_act = arrays_planar( index_object ).parameters.N_elements_axis;
                    rep_act( index_dimension ) = 1;
                    intervals{ index_dimension } = repmat( reshape( math.interval( lbs, ubs ), shape_act ), rep_act );

                end % for index_dimension = 1:arrays_planar( index_object ).N_dimensions

                % create aperture
                apertures{ index_object } = transducers.face_planar_orthotope( intervals{ : } );

            end % for index_object = 1:numel( arrays_planar )

            % avoid cell array for single planar transducer array
            if isscalar( arrays_planar )
                apertures = apertures{ 1 };
            end

        end % function apertures = create_aperture( arrays_planar )

        %------------------------------------------------------------------
        % centers
        %------------------------------------------------------------------
        function objects_out = centers( arrays_planar )

            % initialize output
            objects_out = cell( size( arrays_planar ) );

            % iterate planar transducer arrays
            for index_object = 1:numel( arrays_planar )

                % extract and augment center coordinates
                pos_center = reshape( [ arrays_planar( index_object ).aperture.pos_center ], [ arrays_planar( index_object ).N_dimensions, arrays_planar( index_object ).N_elements ] )';
                objects_out{ index_object } = [ pos_center, zeros( arrays_planar( index_object ).N_elements, 1 ) ];

            end % for index_object = 1:numel( arrays_planar )

            % avoid cell array for single planar transducer array
            if isscalar( arrays_planar )
                objects_out = objects_out{ 1 };
            end

        end % function objects_out = centers( arrays_planar )

        %------------------------------------------------------------------
        % discretize planar transducer array
        %------------------------------------------------------------------
        function structs_out = discretize( arrays_planar, c_avg, options_elements )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % multiple arrays_planar / scalar options_elements
            if ~isscalar( arrays_planar ) && isscalar( options_elements )
                options_elements = repmat( options_elements, size( arrays_planar ) );
            end

            % ensure equal number of dimensions and sizes
            auxiliary.mustBeEqualSize( arrays_planar, c_avg, options_elements );

            %--------------------------------------------------------------
            % 2.) create fields
            %--------------------------------------------------------------
            % specify cell array for structs_out
            structs_out = cell( size( arrays_planar ) );

            % iterate planar transducer arrays
            for index_object = 1:numel( arrays_planar )

                % discretize aperture
                grids_act = discretize( arrays_planar( index_object ).aperture, options_elements( index_object ) );

                % specify cell arrays for apodization and distances
                apodization_act = cell( size( grids_act ) );
                time_delays_act = cell( size( grids_act ) );

                % compute apodization weights
                for index_element = 1:numel( grids_act )

                    % compute normalized relative positions of the grid points
                    positions_rel = grids_act( index_element ).positions - arrays_planar( index_object ).aperture( index_element ).pos_center;
                    positions_rel_norm = 2 * positions_rel ./ arrays_planar( index_object ).parameters.element_width_axis;

                    % compute apodization weights
                    apodization_act{ index_element } = arrays_planar( index_object ).parameters.apodization( positions_rel_norm );

                    % compute time delays for each coordinate axis
                    temp = sum( sqrt( positions_rel.^2 + arrays_planar( index_object ).parameters.axial_focus_axis.^2 ), 2 ) / c_avg( index_object );
                    time_delays_act{ index_element } = max( temp ) - temp;

                end % for index_element = 1:numel( grids_act )

                % create structures
                structs_out{ index_object } = struct( 'grid', num2cell( grids_act ), 'apodization', apodization_act, 'time_delays', time_delays_act );

            end % for index_object = 1:numel( arrays_planar )

            % avoid cell array for single planar transducer array
            if isscalar( arrays_planar )
                structs_out = structs_out{ 1 };
            end

        end % function structs_out = discretize( arrays_planar, c_avg, options_elements )

        %------------------------------------------------------------------
        % inverse index transform
        %------------------------------------------------------------------
        function indices_axis = inverse_index_transform( arrays_planar, indices_linear )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % ensure cell array for indices_linear
            if ~iscell( indices_linear )
                indices_linear = { indices_linear };
            end

            % multiple arrays_planar / single indices_linear
            if ~isscalar( arrays_planar ) && isscalar( indices_linear )
                indices_linear = repmat( indices_linear, size( arrays_planar ) );
            end

            % single arrays_planar / multiple indices_linear
            if isscalar( arrays_planar ) && ~isscalar( indices_linear )
                arrays_planar = repmat( arrays_planar, size( indices_linear ) );
            end

            % ensure equal number of dimensions and sizes
            auxiliary.mustBeEqualSize( arrays_planar, indices_linear );

            %--------------------------------------------------------------
            % 2.) convert linear indices into subscripts
            %--------------------------------------------------------------
            % specify cell array for indices_axis
            indices_axis = cell( size( arrays_planar ) );

            % iterate planar transducer arrays
            for index_array = 1:numel( arrays_planar )

                % convert linear indices into subscripts
                temp = cell( 1, arrays_planar( index_array ).N_dimensions );
                [ temp{ : } ] = ind2sub( arrays_planar( index_array ).parameters.N_elements_axis, indices_linear{ index_array }( : ) );
                indices_axis{ index_array } = cat( 2, temp{ : } );

            end % for index_array = 1:numel( arrays_planar )

            % avoid cell array for single arrays_planar
            if isscalar( arrays_planar )
                indices_axis = indices_axis{ 1 };
            end

        end % function indices_axis = inverse_index_transform( arrays_planar, indices_linear )

	end % methods

end % classdef array_planar < transducers.array
