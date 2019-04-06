%
% superclass for all planar transducer arrays
%
% author: Martin F. Schiffner
% date: 2017-04-19
% modified: 2019-04-02
%
classdef array_planar < transducers.array

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% properties
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	properties (SetAccess = private)

        % dependent properties
        element_pitch_axis ( 1, : ) physical_values.length	% pitches along each coordinate axis (m)
        aperture transducers.face_planar                    % aperture

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
        % create aperture
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

            % avoid cell array for single parameter object
            if numel( arrays_planar ) == 1
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

            % do not return cell array for single object
            if numel( arrays_planar ) == 1
                objects_out = objects_out{ 1 };
            end

        end % function objects_out = centers( arrays_planar )

        %------------------------------------------------------------------
        % discretize planar transducer array
        %------------------------------------------------------------------
        function objects_out = discretize( arrays_planar, options_elements )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % multiple arrays_planar / scalar options_elements
            if ~isscalar( arrays_planar ) && isscalar( options_elements )
                options_elements = repmat( options_elements, size( arrays_planar ) );
            end

            % ensure equal number of dimensions and sizes
            auxiliary.mustBeEqualSize( arrays_planar, options_elements );

            %--------------------------------------------------------------
            % 2.) create regular grids
            %--------------------------------------------------------------
            % specify cell array for objects_out
            objects_out = cell( size( arrays_planar ) );

            % iterate planar transducer arrays
            for index_object = 1:numel( arrays_planar )

                % discretize aperture
                objects_out{ index_object } = discretize( arrays_planar( index_object ).aperture, options_elements( index_object ) );

            end % for index_object = 1:numel( arrays_planar )

            % do not return cell array for single planar transducer array
            if numel( arrays_planar ) == 1
                objects_out = objects_out{ 1 };
            end

        end % function objects_out = discretize( arrays_planar, options_elements )

	end % methods

end % classdef array_planar < transducers.array
