%
% superclass for all Cartesian coordinates
%
% author: Martin F. Schiffner
% date: 2019-03-22
% modified: 2019-03-22
%
classdef coordinates_cartesian < coordinates.coordinates_affine

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % methods
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	methods

        %------------------------------------------------------------------
        % constructor
        %------------------------------------------------------------------
        function objects = coordinates_cartesian( varargin )

            %--------------------------------------------------------------
            % 1.) constructor of superclass
            %--------------------------------------------------------------
            objects@coordinates.coordinates_affine( varargin{ : } );

        end % function objects = coordinates_cartesian( varargin )

    end % methods

end % classdef coordinates_cartesian < coordinates.coordinates_affine
