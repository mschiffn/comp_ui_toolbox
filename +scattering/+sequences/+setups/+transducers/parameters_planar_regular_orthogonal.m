%
% superclass for all planar transducer array parameters
%
% author: Martin F. Schiffner
% date: 2019-02-18
% modified: 2019-04-11
%
classdef parameters_planar_regular_orthogonal < transducers.parameters_planar_regular

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% properties
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	properties (SetAccess = private)

        % independent geometrical properties
        element_width_axis ( 1, : ) physical_values.length	% widths of the vibrating faces along each coordinate axis (m)
        element_kerf_axis ( 1, : ) physical_values.length	% widths of the kerfs separating the adjacent elements along each coordinate axis (m)

    end % properties

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% methods
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    methods

        %------------------------------------------------------------------
        % constructor
        %------------------------------------------------------------------
        function objects = parameters_planar_regular_orthogonal( N_elements_axis, element_width_axis, element_kerf_axis, apodization, axial_focus_axis, str_model, str_vendor )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % ensure cell array for N_elements_axis
            if ~iscell( N_elements_axis )
                N_elements_axis = { N_elements_axis };
            end

            % ensure cell array for element_width_axis
            if ~iscell( element_width_axis )
                element_width_axis = { element_width_axis };
            end

            % ensure cell array for element_kerf_axis
            if ~iscell( element_kerf_axis )
                element_kerf_axis = { element_kerf_axis };
            end

            % ensure equal number of dimensions and sizes of cell arrays
            auxiliary.mustBeEqualSize( N_elements_axis, element_width_axis, element_kerf_axis );

            %--------------------------------------------------------------
            % 2.) constructor of superclass
            %--------------------------------------------------------------
            objects@transducers.parameters( N_elements_axis, apodization, axial_focus_axis, str_model, str_vendor );

            %--------------------------------------------------------------
            % 3.) set parameters of planar transducer array
            %--------------------------------------------------------------
            for index_object = 1:numel( objects )

                % ensure equal number of dimensions and sizes
                auxiliary.mustBeEqualSize( objects( index_object ).N_elements_axis, element_width_axis{ index_object }, element_kerf_axis{ index_object } );

                % set independent properties
                objects( index_object ).element_width_axis = element_width_axis{ index_object };
                objects( index_object ).element_kerf_axis = element_kerf_axis{ index_object };

            end % for index_object = 1:numel( objects )

        end % function objects = parameters_planar_regular_orthogonal( N_elements_axis, element_width_axis, element_kerf_axis, apodization, axial_focus_axis, str_model, str_vendor )

	end % methods

end % classdef parameters_planar_regular_orthogonal < transducers.parameters
