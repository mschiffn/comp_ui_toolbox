%
% superclass for all coordinates
%
% author: Martin F. Schiffner
% date: 2019-03-22
% modified: 2019-03-25
%
classdef coordinates

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % methods
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	methods

        %------------------------------------------------------------------
        % constructor
        %------------------------------------------------------------------
        function objects = coordinates( size_coordinates )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % return for no input arguments
            if nargin == 0
                return;
            end

            %--------------------------------------------------------------
            % 2.) create coordinates
            %--------------------------------------------------------------
            objects = repmat( objects, size_coordinates );

        end % function objects = coordinates( size_coordinates )

    end % methods

end % classdef coordinates
