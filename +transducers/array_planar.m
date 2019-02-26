%
% superclass for all planar transducer arrays
%
% author: Martin F. Schiffner
% date: 2017-04-19
% modified: 2019-02-18
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
        function objects = array_planar( N_dimensions, parameters_planar )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % ensure class transducers.parameters_planar
            if ~isa( parameters_planar, 'transducers.parameters_planar' )
                errorStruct.message     = 'parameters_planar must be transducers.parameters_planar!';
                errorStruct.identifier	= 'array_planar:NoParametersPlanar';
                error( errorStruct );
            end
            % assertion: parameters_planar is transducers.parameters_planar

            %--------------------------------------------------------------
            % 2.) constructor of superclass
            %--------------------------------------------------------------
            objects@transducers.array( N_dimensions, parameters_planar );

            %--------------------------------------------------------------
            % 3.) create planar transducer arrays
            %--------------------------------------------------------------
            for index_object = 1:numel( objects )

                %----------------------------------------------------------
                % a) set independent properties
                %----------------------------------------------------------
                objects( index_object ).element_width_axis = parameters_planar( index_object ).element_width_axis( 1:objects( index_object ).N_dimensions );
                objects( index_object ).element_kerf_axis = parameters_planar( index_object ).element_kerf_axis( 1:objects( index_object ).N_dimensions );
                % assertion: independent geometrical properties specify valid planar transducer array

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

        end % function objects = array_planar( N_dimensions, parameters_planar )

        %------------------------------------------------------------------
        % centers
        %------------------------------------------------------------------
        function objects_out = centers( objects_in )

            % initialize output
            objects_out = cell( size( objects_in ) );

            % iterate objects
            for index_object = 1:numel( objects_in )

                % compute center coordinates
                temp = center( objects_in( index_object ).aperture );
                pos_center = physical_values.position( zeros( numel( objects_in( index_object ).aperture ), objects_in( index_object ).N_dimensions + 1 ) );
                pos_center( :, 1 ) = [ temp{ : } ];
                objects_out{ index_object } = pos_center;

            end % for index_object = 1:numel( objects_in )

            % do not return cell array for single object
            if numel( objects_in ) == 1
                objects_out = objects_out{ 1 };
            end

        end % function objects_out = centers( objects_in )

        %------------------------------------------------------------------
        % discretize planar transducer array
        %------------------------------------------------------------------
        function objects_out = discretize( objects_in, N_interp_axis )

            objects_out = cell( size( objects_in ) );

            for index_object = 1:numel( objects_in )
                % TODO: check dimensions
                delta_axis = objects_in( index_object ).element_width_axis ./ N_interp_axis;
                objects_out{ index_object } = discretize( objects_in( index_object ).aperture, delta_axis );

            end % for index_object = 1:numel( objects_in )

            % do not return cell array for single object
            if numel( objects_in ) == 1
                objects_out = objects_out{ 1 };
            end

        end % function objects_out = discretize( objects_in, N_interp_axis )

	end % methods

end % classdef array_planar < transducers.array
