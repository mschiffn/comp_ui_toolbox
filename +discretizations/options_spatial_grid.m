%
% superclass for all grid options
%
% author: Martin F. Schiffner
% date: 2019-02-19
% modified: 2019-05-23
%
classdef options_spatial_grid < discretizations.options_spatial

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% properties
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	properties (SetAccess = private)

        % independent properties
        options_elements ( 1, : ) discretizations.parameters = discretizations.parameters_number
        options_FOV ( 1, 1 ) discretizations.parameters = discretizations.parameters_distance

    end % properties

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% methods
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	methods

        %------------------------------------------------------------------
        % constructor
        %------------------------------------------------------------------
        function objects = options_spatial_grid( options_elements, options_FOV )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % specify default values
            if nargin == 0
                options_elements = discretizations.parameters_number;
                options_FOV = discretizations.parameters_distance;
            end

            % ensure cell array for options_elements
            if ~iscell( options_elements )
                options_elements = { options_elements };
            end

            % ensure equal number of dimensions and sizes
            auxiliary.mustBeEqualSize( options_elements, options_FOV );

            %--------------------------------------------------------------
            % 2.) constructor of superclass
            %--------------------------------------------------------------
            objects@discretizations.options_spatial( options_elements );

            %--------------------------------------------------------------
            % 3.) set independent properties
            %--------------------------------------------------------------
            for index_object = 1:numel( objects )

                objects( index_object ).options_elements = options_elements{ index_object };
                objects( index_object ).options_FOV = options_FOV( index_object );

            end % for index_object = 1:numel( objects )

        end % function objects = options_spatial_grid( options_elements, options_FOV )

	end % methods

end % classdef options_spatial_grid < discretizations.options_spatial
