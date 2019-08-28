% abstract superclass for all spatial discretization methods using grids
%
% author: Martin F. Schiffner
% date: 2019-08-20
% modified: 2019-08-20
%
classdef options_spatial_method_grid_numbers < discretizations.options_spatial_method_grid

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % properties
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	properties (SetAccess = private)

        % independent properties
        numbers ( :, 1 ) double { mustBeInteger, mustBePositive, mustBeNonempty } = [ 4; 53 ];

    end % properties

	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	%% methods
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    methods

        %------------------------------------------------------------------
        % constructor
        %------------------------------------------------------------------
        function objects = options_spatial_method_grid_numbers( numbers )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % specify default numbers if no arguments
            if nargin == 0
                numbers = [ 4; 53 ];
            end

            % ensure cell array for numbers
            if ~iscell( numbers )
                numbers = { numbers };
            end

            %--------------------------------------------------------------
            % 2.) create spatial discretization method options using grids
            %--------------------------------------------------------------
            % constructor of superclass
            objects@discretizations.options_spatial_method_grid( size( numbers ) );

            % iterate spatial discretization method options using grids
            for index_object = 1:numel( objects )

                % set independent properties
                objects( index_object ).numbers = numbers{ index_object };

            end % for index_object = 1:numel( objects )

        end % function objects = options_spatial_method_grid_numbers( numbers )

	end % methods

end % classdef options_spatial_method_grid_numbers < discretizations.options_spatial_method_grid
