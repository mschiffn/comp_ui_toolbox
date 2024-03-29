% abstract superclass for all spatial discretization methods using grids
%
% author: Martin F. Schiffner
% date: 2019-08-21
% modified: 2019-10-21
%
classdef grid_distances < scattering.sequences.setups.discretizations.methods.grid

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % properties
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	properties (SetAccess = private)

        % independent properties
        distances ( :, 1 ) physical_values.length { mustBeNonempty } = physical_values.meter( 76.2e-6 * ones(3, 1) )

    end % properties

	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	%% methods
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    methods

        %------------------------------------------------------------------
        % constructor
        %------------------------------------------------------------------
        function objects = grid_distances( distances )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % specify default distances if no arguments
            if nargin == 0
                distances = physical_values.meter( 76.2e-6 * ones(3, 1) );
            end

            % ensure cell array for distances
            if ~iscell( distances )
                distances = { distances };
            end

            %--------------------------------------------------------------
            % 2.) create spatial discretization method options using grids
            %--------------------------------------------------------------
            % constructor of superclass
            objects@scattering.sequences.setups.discretizations.methods.grid( size( distances ) );

            % iterate spatial discretization method options using grids
            for index_object = 1:numel( objects )

                % set independent properties
                objects( index_object ).distances = distances{ index_object };

            end % for index_object = 1:numel( objects )

        end % function objects = grid_distances( distances )

	end % methods

end % classdef grid_distances < scattering.sequences.setups.discretizations.methods.grid
