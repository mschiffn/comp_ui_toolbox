%
% superclass for all continuous planar vibrating faces
%
% author: Martin F. Schiffner
% date: 2019-02-18
% modified: 2019-08-20
%
classdef face_planar < transducers.face

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% methods
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	methods

        %------------------------------------------------------------------
        % constructor
        %------------------------------------------------------------------
        function objects = face_planar( varargin )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------

            %--------------------------------------------------------------
            % 2.) constructor of superclass
            %--------------------------------------------------------------
            objects@transducers.face( varargin{ : } );

        end % function objects = face_planar( varargin )

    end % methods

end % classdef face_planar < transducers.continuous.face & transducers.face_planar
