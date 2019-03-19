%
% superclass for all discretization parameters
%
% author: Martin F. Schiffner
% date: 2019-02-15
% modified: 2019-03-19
%
classdef parameters

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % methods
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    methods

        %------------------------------------------------------------------
        % constructor
        %------------------------------------------------------------------
        function objects = parameters( varargin )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % check number of arguments
            if nargin ~= 1
                return;
            end

            %--------------------------------------------------------------
            % 2.) create discretization parameters
            %--------------------------------------------------------------
            objects = repmat( objects, size( varargin{1} ) );

        end % function objects = parameters( varargin )

	end % methods

end % classdef parameters
