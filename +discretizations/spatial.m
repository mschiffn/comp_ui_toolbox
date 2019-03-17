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

        %------------------------------------------------------------------
        % compute hash value
        %------------------------------------------------------------------
        function str_hash = hash( object )

            % use DataHash function to compute hash value
            str_hash = auxiliary.DataHash( object );

        end % function str_hash = hash( object )

	end % methods

end % classdef spatial
