%
% superclass for all spatial discretization options
%
% author: Martin F. Schiffner
% date: 2019-02-20
% modified: 2019-03-19
%
classdef options_spatial

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % methods
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    methods

        %------------------------------------------------------------------
        % constructor
        %------------------------------------------------------------------
        function objects = options_spatial( varargin )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            if nargin == 0
                return;
            end

            %--------------------------------------------------------------
            % 2.) set independent properties
            %--------------------------------------------------------------
            objects = repmat( objects, size( varargin{ 1 } ) );

        end % function objects = options_spatial( varargin )

	end % methods

end % classdef options_spatial
