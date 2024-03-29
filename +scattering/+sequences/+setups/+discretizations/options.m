%
% superclass for all grid options
%
% author: Martin F. Schiffner
% date: 2019-02-19
% modified: 2019-10-21
%
classdef options

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% properties
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	properties (SetAccess = private)

        % independent properties
        method_faces ( 1, 1 ) scattering.sequences.setups.discretizations.methods.method = scattering.sequences.setups.discretizations.methods.grid_numbers
        method_FOV ( 1, 1 ) scattering.sequences.setups.discretizations.methods.method = scattering.sequences.setups.discretizations.methods.grid_distances

    end % properties

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% methods
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	methods

        %------------------------------------------------------------------
        % constructor
        %------------------------------------------------------------------
        function objects = options( methods_faces, methods_FOV )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % specify default values
            if nargin == 0
                methods_faces = scattering.sequences.setups.discretizations.methods.grid_numbers;
                methods_FOV = scattering.sequences.setups.discretizations.methods.grid_distances;
            end

            % ensure cell array for methods_faces
            if ~iscell( methods_faces )
                methods_faces = { methods_faces };
            end

            % ensure equal number of dimensions and sizes
            auxiliary.mustBeEqualSize( methods_faces, methods_FOV );

            %--------------------------------------------------------------
            % 2.) create grid options
            %--------------------------------------------------------------
            % repeat default discretization options
            objects = repmat( objects, size( methods_faces ) );

            % iterate grid options
            for index_object = 1:numel( objects )

                % set independent properties
                objects( index_object ).method_faces = methods_faces{ index_object };
                objects( index_object ).method_FOV = methods_FOV( index_object );

            end % for index_object = 1:numel( objects )

        end % function objects = options( methods_faces, methods_FOV )

	end % methods

end % classdef options
