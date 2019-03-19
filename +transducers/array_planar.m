%
% superclass for all planar transducer arrays
%
% author: Martin F. Schiffner
% date: 2017-04-19
% modified: 2019-03-18
%
classdef array_planar < transducers.array

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % properties
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	properties (SetAccess = private)

        % independent properties
        element_width_axis ( 1, : ) double { mustBeReal, mustBePositive, mustBeFinite }     % widths of the vibrating faces along each coordinate axis (m)
        element_kerf_axis ( 1, : ) double { mustBeReal, mustBeNonnegative, mustBeFinite }	% widths of the kerfs separating the adjacent elements along each coordinate axis (m)

        % dependent properties
        element_pitch_axis ( 1, : ) double { mustBeReal, mustBePositive, mustBeFinite }     % pitches along each coordinate axis (m)
        aperture transducers.face_planar                                                    % aperture

	end % properties

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % methods
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
            % 3.) create planar transducer arrays
            %--------------------------------------------------------------
            for index_object = 1:numel( objects )

                %----------------------------------------------------------
                % a) set independent properties
                %----------------------------------------------------------
                objects( index_object ).element_width_axis = parameters_planar( index_object ).element_width_axis( 1:objects( index_object ).N_dimensions );
                objects( index_object ).element_kerf_axis = parameters_planar( index_object ).element_kerf_axis( 1:objects( index_object ).N_dimensions );

                %----------------------------------------------------------
                % b) set dependent properties
                %----------------------------------------------------------
                objects( index_object ).element_pitch_axis = objects( index_object ).element_width_axis + objects( index_object ).element_kerf_axis;

                % create position intervals for aperture
                M_elements_axis = ( objects( index_object ).N_elements_axis - 1 ) / 2;
                intervals = cell( 1, objects( index_object ).N_dimensions );

                for index_dimension = 1:objects( index_object ).N_dimensions

                    % compute lower and upper bounds on the position intervals on current axis
                    pos_ctr = ( -M_elements_axis( index_dimension ):M_elements_axis( index_dimension ) ) * objects( index_object ).element_pitch_axis( index_dimension );
                    pos_rel = objects( index_object ).element_width_axis( index_dimension ) / 2;
                    lbs = physical_values.position( pos_ctr - pos_rel );
                    ubs = physical_values.position( pos_ctr + pos_rel );

                    % create position intervals
                    shape_act = ones( 1, max( [ objects( index_object ).N_dimensions, 2 ] ) );
                    shape_act( index_dimension ) = objects( index_object ).N_elements_axis( index_dimension );
                    rep_act = objects( index_object ).N_elements_axis;
                    rep_act( index_dimension ) = 1;
                    intervals{ index_dimension } = repmat( reshape( physical_values.interval_position( lbs, ubs ), shape_act ), rep_act );

                end % for index_dimension = 1:objects( index_object ).N_dimensions

                % create aperture
                objects( index_object ).aperture = transducers.face_planar_orthotope( intervals{ : } );

            end % for index_object = 1:numel( objects )

        end % function objects = array_planar( parameters_planar, N_dimensions )

        %------------------------------------------------------------------
        % centers
        %------------------------------------------------------------------
        function objects_out = centers( arrays_planar )

            % initialize output
            objects_out = cell( size( arrays_planar ) );

            % iterate planar transducer arrays
            for index_object = 1:numel( arrays_planar )

                % initialize results
                objects_out{ index_object } = physical_values.position( zeros( arrays_planar( index_object ).N_elements, arrays_planar( index_object ).N_dimensions + 1 ) );

                % extract and augment center coordinates
                pos_center = reshape( [ arrays_planar( index_object ).aperture.pos_center ], [ arrays_planar( index_object ).N_dimensions, arrays_planar( index_object ).N_elements ] )';
                objects_out{ index_object }( :, 1:arrays_planar( index_object ).N_dimensions ) = pos_center;

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
            % ensure cell array for options_elements
            if ~iscell( options_elements )
                options_elements = { options_elements };
            end

            % multiple arrays_planar / single options_elements
            if ~isscalar( arrays_planar ) && isscalar( options_elements )
                options_elements = repmat( options_elements, size( arrays_planar ) );
            end

            % ensure equal number of dimensions and sizes
            auxiliary.mustBeEqualSize( arrays_planar, options_elements );

            %--------------------------------------------------------------
            % 2.) create regular grids
            %--------------------------------------------------------------
            % initialize output
            objects_out = cell( size( arrays_planar ) );

            % iterate planar transducer arrays
            for index_object = 1:numel( arrays_planar )

                % discretize aperture
                objects_out{ index_object } = discretize( arrays_planar( index_object ).aperture, options_elements{ index_object } );

            end % for index_object = 1:numel( arrays_planar )

            % do not return cell array for single object
            if numel( arrays_planar ) == 1
                objects_out = objects_out{ 1 };
            end

        end % function objects_out = discretize( arrays_planar, options_elements )

	end % methods

end % classdef array_planar < transducers.array
