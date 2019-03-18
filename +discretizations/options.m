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
        spatial ( 1, 1 ) discretizations.options_spatial = discretizations.options_spatial_grid         % spatial discretization
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
            % return if no input argument
            if nargin == 0
                return;
            end

            % ensure equal number of dimensions and sizes
            auxiliary.mustBeEqualSize( spatial, spectral );

            %--------------------------------------------------------------
            % 2.) create spatiospectral discretization options
            %--------------------------------------------------------------
            % create objects
            objects = repmat( objects, size( spatial ) );

            for index_object = 1:numel( spatial )

                % set independent properties
                objects( index_object ).spatial = spatial( index_object );
                objects( index_object ).spectral = spectral( index_object );

            end % for index_object = 1:numel( spatial )

        end % function objects = options( spatial, spectral )

	end % methods

end % classdef options
