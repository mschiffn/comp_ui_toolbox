%
% superclass for all planar vibrating faces
% author: Martin F. Schiffner
% date: 2019-02-18
% modified: 2019-06-03
%
classdef face_planar < transducers.face

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% properties
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	properties (SetAccess = private)

        % independent properties
        
    end % properties

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% methods
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    methods

        %------------------------------------------------------------------
        % constructor
        %------------------------------------------------------------------
        function objects = face_planar( varargin )

            %--------------------------------------------------------------
            % constructor of superclass
            %--------------------------------------------------------------
            objects@transducers.face();
%             objects.pos_center = center( objects );
        end

    end % methods

end % classdef face_planar < transducers.face
