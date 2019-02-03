%
% superclass for all planar transducer arrays
%
% author: Martin F. Schiffner
% date: 2017-04-19
% modified: 2019-01-17
%
classdef planar_transducer_array < transducer_models.transducer_array

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % properties
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	properties (SetAccess = private)

        % independent geometrical properties
        element_width_axis ( 1, : ) double { mustBeReal, mustBePositive, mustBeFinite }     % widths of the vibrating faces along each coordinate axis (m)
        element_kerf_axis ( 1, : ) double { mustBeReal, mustBeNonnegative, mustBeFinite }	% widths of the kerfs separating the adjacent elements along each coordinate axis (m)

        % independent electromechanical properties

        % dependent geometrical properties
        element_pitch_axis ( 1, : ) double { mustBeReal, mustBePositive, mustBeFinite }     % pitches along each coordinate axis (m)
        width_axis ( 1, : ) double { mustBeReal, mustBePositive, mustBeFinite }             % width of aperture along each coordinate axis (m)
        grid_ctr ( 1, 1 ) discretization.grid                                                        % regular grid of center coordinates of the vibrating faces

        % dependent discretization properties
        N_points ( 1, 1 ) double { mustBeInteger, mustBePositive } = 10                     % total number of grid points (1)
        % TODO: grid_array?
        grid ( :, 1 ) discretization.grid                                                            % regular grid of points on each vibrating face
    end % properties

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % methods
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    methods

        %------------------------------------------------------------------
        % constructor
        %------------------------------------------------------------------
        function obj = planar_transducer_array( N_elements_axis, element_width_axis, element_kerf_axis, str_name )

            % constructor of superclass
            obj@transducer_models.transducer_array( N_elements_axis(:)', str_name );

            % check number of dimensions
            if numel( element_width_axis ) ~= obj.N_dimensions || numel( element_kerf_axis ) ~= obj.N_dimensions
                errorStruct.message     = 'Number of components in element_width_axis and element_kerf_axis must match N_dimensions!';
                errorStruct.identifier	= 'planar_transducer_array:DimensionMismatch';
                error( errorStruct );
            end

            % set independent geometrical properties
            obj.element_width_axis = element_width_axis(:)';
            obj.element_kerf_axis = element_kerf_axis(:)';
            % assertion: independent geometrical properties specify valid planar transducer array

            % dependent geometrical properties
            obj.element_pitch_axis = obj.element_width_axis + obj.element_kerf_axis;
            obj.width_axis = obj.N_elements_axis .* obj.element_pitch_axis - obj.element_kerf_axis;

            % regular grid of center coordinates of the vibrating faces
            offset_axis	= ( 1 - obj.N_elements_axis ) .* obj.element_pitch_axis / 2;
            obj.grid_ctr = discretization.grid( obj.N_elements_axis, obj.element_pitch_axis, offset_axis );
        end

        %------------------------------------------------------------------
        % discretize planar transducer array
        %------------------------------------------------------------------
        function obj = discretize( obj, N_interp_axis )

            % check number of dimensions
            N_interp_axis = N_interp_axis(:)';
            if numel( N_interp_axis ) ~= obj.N_dimensions
                errorStruct.message     = 'Number of components in N_interp_axis must match N_dimensions!';
                errorStruct.identifier	= 'discretize:DimensionMismatch';
                error( errorStruct );
            end

            % grid parameters for each array element
            obj.N_points = obj.N_elements * prod( N_interp_axis, 2 );
            delta_axis = obj.element_width_axis ./ N_interp_axis;
            offset_axis_const = ( 1 - N_interp_axis ) .* delta_axis / 2;

            % create grid for each array element
            for index_element = 1:obj.N_elements

                offset_axis = obj.grid_ctr.positions( index_element, :) + offset_axis_const;
                obj.grid( index_element ) = discretization.grid( N_interp_axis, delta_axis, offset_axis );
            end
        end

        %------------------------------------------------------------------
        % return positions
        %------------------------------------------------------------------
        function positions = positions( obj, index_dimension )

            % read positions of grid points
            positions = [ obj.grid.positions ];
            positions = positions( :, index_dimension:obj.N_dimensions:end );
            positions = positions( : );
        end
	end % methods

end % classdef planar_transducer_array < transducer_models.transducer_array
