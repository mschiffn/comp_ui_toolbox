%
% superclass for all vibrating faces
% author: Martin F. Schiffner
% date: 2019-02-18
% modified: 2019-02-18
%
classdef face

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % methods
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    methods

        %------------------------------------------------------------------
        % constructor
        %------------------------------------------------------------------
        function objects = face( varargin )

            if nargin == 0
                return;
            end

            objects = repmat( objects, size( varargin{ 1 } ) );
        end % function objects = face( varargin )

    end % methods

end % classdef face
