%
% superclass for all grid options
%
% author: Martin F. Schiffner
% date: 2019-02-19
% modified: 2019-03-18
%
classdef options_spatial_grid < discretizations.options_spatial

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % properties
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	properties (SetAccess = private)

        % independent properties
        grids_elements_options ( 1, : )
        grid_FOV_options ( 1, 1 )

        
        N_points_per_element_axis ( 1, : ) { mustBeInteger, mustBeNonempty } = [ 4, 10 ];
        delta_axis = 3.0480e-04 * ones(1,3) / 4;

    end % properties

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % methods
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    methods

        %------------------------------------------------------------------
        % constructor
        %------------------------------------------------------------------
        function object = options_spatial_grid( varargin )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % check number of arguments
            if nargin ~= 1
                return;
            end

            
            % ensure class discretizations.method
            if ~isa( method, 'discretizations.method' ) || numel( method ) ~= 1
                errorStruct.message     = 'method must be a single discretizations.method!';
                errorStruct.identifier	= 'scattering_operator:NoSingleMethod';
                error( errorStruct );
            end
            % assertion: method is a single discretizations.method

            %--------------------------------------------------------------
            % 2.) set independent properties
            %--------------------------------------------------------------

        end % function object = options_spatial_grid( varargin )

	end % methods

end % classdef options_spatial_grid
