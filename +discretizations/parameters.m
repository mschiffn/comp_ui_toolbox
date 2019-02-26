%
% superclass for all discretization parameters
%
% author: Martin F. Schiffner
% date: 2019-02-15
% modified: 2019-02-15
%
classdef parameters

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
        function object = parameters( varargin )

            % check number of arguments
            if nargin ~= 1
                return;
            end

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------

            %--------------------------------------------------------------
            % 2.) set independent properties
            %--------------------------------------------------------------

        end % function object = parameters( varargin )

	end % methods

end % classdef parameters
