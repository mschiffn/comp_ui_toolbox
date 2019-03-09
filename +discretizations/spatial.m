%
% superclass for all spatial discretizations
%
% author: Martin F. Schiffner
% date: 2019-02-17
% modified: 2019-02-17
%
classdef spatial

	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	% properties
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	properties (SetAccess = private)
    end % properties

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % methods
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	methods

        %------------------------------------------------------------------
        % constructor
        %------------------------------------------------------------------
        function objects = spatial( )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % return empty object if no arguments
            if nargin == 0
                return;
            end

        end % function objects = spatial( )

	end % methods

end % classdef spatial
