%
% superclass for all grid parameters
%
% author: Martin F. Schiffner
% date: 2019-02-15
% modified: 2019-02-19
%
classdef parameters_grid < discretizations.parameters

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % properties
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	properties (SetAccess = private)

        % independent properties
        N_points_axis ( 1, : ) double { mustBeInteger, mustBePositive, mustBeNonempty } = [ 128, 128 ]	% numbers of grid points along each coordinate axis (1)
        delta_axis ( 1, : ) double { mustBeReal, mustBePositive, mustBeNonempty } = [ 1e-4, 1e-4 ]      % constant spacings between the adjacent grid points along each coordinate axis (m)
        offset_axis ( 1, : ) double { mustBeReal, mustBeNonempty } = [ -63.5e-4, 0.5e-4]                % arbitrary offset (m)

    end % properties

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % methods
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    methods

        %------------------------------------------------------------------
        % constructor
        %------------------------------------------------------------------
        function objects = parameters_grid( varargin )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------

            %--------------------------------------------------------------
            % 2.) constructor of superclass
            %--------------------------------------------------------------
            objects@discretizations.parameters( varargin{ : } );

            %--------------------------------------------------------------
            % 3.) set independent properties
            %--------------------------------------------------------------
            objects.N_points_axis = N_points_axis;
            objects.delta_axis = N_points_axis;
            objects.offset_axis = N_points_axis;
            
        end % function object = parameters_grid( varargin )

	end % methods

end % classdef parameters_grid
