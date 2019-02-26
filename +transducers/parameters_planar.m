%
% superclass for all planar transducer array parameters
%
% author: Martin F. Schiffner
% date: 2019-02-18
% modified: 2019-02-18
%
classdef parameters_planar < transducers.parameters

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % properties
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	properties (SetAccess = private)

        % independent geometrical properties
        element_width_axis ( 1, : ) double { mustBeReal, mustBePositive, mustBeFinite }     % widths of the vibrating faces along each coordinate axis (m)
        element_kerf_axis ( 1, : ) double { mustBeReal, mustBeNonnegative, mustBeFinite }	% widths of the kerfs separating the adjacent elements along each coordinate axis (m)

    end % properties

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % methods
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    methods

        %------------------------------------------------------------------
        % constructor
        %------------------------------------------------------------------
        function objects = parameters_planar( N_elements_axis, element_width_axis, element_kerf_axis, str_model, str_vendor )

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

            % ensure cell array for str_model
            if ~iscell( str_model )
                str_model = { str_model };
            end

            % ensure cell array for str_vendor
            if ~iscell( str_vendor )
                str_vendor = { str_vendor };
            end

            % ensure equal number of dimensions and sizes of cell arrays
            auxiliary.mustBeEqualSize( N_elements_axis, element_width_axis, element_kerf_axis, str_model, str_vendor );
            % assertion: N_elements_axis, element_width_axis, element_kerf_axis, and str_name have equal sizes

            %--------------------------------------------------------------
            % 2.) constructor of superclass
            %--------------------------------------------------------------
            objects@transducers.parameters( N_elements_axis, str_model, str_vendor );

            %--------------------------------------------------------------
            % 3.) set parameters of planar transducer array
            %--------------------------------------------------------------
            for index_object = 1:numel( objects )

                % ensure equal number of dimensions and sizes
                auxiliary.mustBeEqualSize( objects( index_object ).N_elements_axis, element_width_axis{ index_object }, element_kerf_axis{ index_object } );
                % assertion: N_elements_axis, element_width_axis, element_kerf_axis, and str_name have equal sizes

                % set independent properties
                objects( index_object ).element_width_axis = element_width_axis{ index_object };
                objects( index_object ).element_kerf_axis = element_kerf_axis{ index_object };

            end % for index_object = 1:numel( objects )

        end % function objects = parameters_planar( N_elements_axis, str_name )

	end % methods

end % classdef parameters_planar
