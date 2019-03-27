%
% superclass for all array elements with orthotope shape
% author: Martin F. Schiffner
% date: 2019-02-18
% modified: 2019-03-25
%
classdef face_planar_orthotope < transducers.face_planar & physical_values.orthotope

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % properties
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	properties (SetAccess = private)

        % independent properties
        pos_center ( 1, : ) physical_values.length	% center coordinates of vibrating face (m)

    end % properties

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % methods
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    methods

        %------------------------------------------------------------------
        % constructor
        %------------------------------------------------------------------
        function objects = face_planar_orthotope( varargin )

            %--------------------------------------------------------------
            % 1.) constructors of superclasses
            %--------------------------------------------------------------
            objects@transducers.face_planar();
            objects@physical_values.orthotope( varargin{ : } );

            for index_object = 1:numel( objects )
                objects( index_object ).pos_center = center( objects( index_object ) );
            end

        end % function objects = face_planar_orthotope( varargin )

    end % methods

end % classdef face_planar_orthotope
