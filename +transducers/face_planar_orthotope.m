%
% superclass for all planar vibrating faces with orthotope shape
%
% author: Martin F. Schiffner
% date: 2019-02-18
% modified: 2019-06-04
%
classdef face_planar_orthotope < transducers.face_planar

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% properties
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	properties (SetAccess = private)

    end % properties

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% methods
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    methods

        %------------------------------------------------------------------
        % constructor
        %------------------------------------------------------------------
        function objects = face_planar_orthotope( apodizations, lenses, varargin )

            %--------------------------------------------------------------
            % 1.) create planar vibrating faces with orthotope shape
            %--------------------------------------------------------------
            % create orthotopes
            orthotopes = math.orthotope( varargin{ : } );

            % constructor of superclass
            objects@transducers.face_planar( orthotopes, apodizations, lenses );

        end % function objects = face_planar_orthotope( apodizations, lenses, varargin )

    end % methods

end % classdef face_planar_orthotope < transducers.face_planar
