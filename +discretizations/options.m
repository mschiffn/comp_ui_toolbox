%
% superclass for all spatiospectral discretization options
%
% author: Martin F. Schiffner
% date: 2019-02-25
% modified: 2019-03-04
%
classdef options

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % properties
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	properties (SetAccess = private)

        % independent properties
        spatial ( 1, 1 ) discretizations.options_spatial_grid = discretizations.options_spatial_grid	% spatial discretization
        spectral ( 1, 1 ) discretizations.options_spectral = discretizations.options_spectral.signal	% spectral discretization

    end % properties

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % methods
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    methods

        %------------------------------------------------------------------
        % constructor
        %------------------------------------------------------------------
        function objects = options( spatial, spectral )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            if nargin == 0
                return;
            end

            %--------------------------------------------------------------
            % 2.) set independent properties
            %--------------------------------------------------------------
            objects.spatial = spatial;
            objects.spectral = spectral;

        end % function objects = options( spatial, spectral )

	end % methods

end % classdef options
