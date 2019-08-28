%
% superclass for all planar vibrating faces with orthotope shape
%
% author: Martin F. Schiffner
% date: 2019-02-18
% modified: 2019-08-22
%
classdef face_planar_orthotope < transducers.face_planar

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% methods
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    methods

        %------------------------------------------------------------------
        % constructor
        %------------------------------------------------------------------
        function objects = face_planar_orthotope( apodizations, lenses, varargin )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % superclass ensures class function_handle for apodizations
            % superclass ensures class transducers.lens for lenses
            % class math.orthotope ensures intervals for varargin

            %--------------------------------------------------------------
            % 2.) create planar vibrating faces with orthotope shape
            %--------------------------------------------------------------
            % create orthotopes
            orthotopes = geometry.orthotope( varargin{ : } );

            % constructor of superclass
            objects@transducers.face_planar( orthotopes, apodizations, lenses );

        end % function objects = face_planar_orthotope( apodizations, lenses, varargin )

    end % methods

end % classdef face_planar_orthotope < transducers.face_planar
