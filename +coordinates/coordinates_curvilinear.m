%
% superclass for all curvilinear coordinates
%
% author: Martin F. Schiffner
% date: 2019-03-22
% modified: 2019-03-22
%
classdef coordinates_curvilinear < coordinates.coordinates

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % methods
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	methods

        %------------------------------------------------------------------
        % constructor
        %------------------------------------------------------------------
        function objects = coordinates_curvilinear( values )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % return for no input arguments
            if nargin == 0
                values = 1;
            end

            % ensure cell array for values
            if ~iscell( values )
                values = { values };
            end

            %--------------------------------------------------------------
            % 2.) constructor of superclass
            %--------------------------------------------------------------
            objects@coordinates.coordinates( size( values ) );

        end % function objects = coordinates_curvilinear( values )

    end % methods

end % classdef coordinates_curvilinear < coordinates.coordinates
