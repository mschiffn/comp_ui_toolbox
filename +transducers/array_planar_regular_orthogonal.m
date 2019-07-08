%
% superclass for all regular orthogonal planar transducer arrays
%
% author: Martin F. Schiffner
% date: 2019-06-04
% modified: 2019-06-04
%
classdef array_planar_regular_orthogonal < transducers.array_planar_regular

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% methods
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	methods

        %------------------------------------------------------------------
        % constructor
        %------------------------------------------------------------------
        function objects = array_planar_regular_orthogonal( faces_ref, element_pitch_axis, N_elements_axis )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % ensure class transducers.face_planar_orthotope
            if ~isa( faces_ref, 'transducers.face_planar_orthotope' )
                errorStruct.message = 'faces_ref must be transducers.face_planar_orthotope!';
                errorStruct.identifier = 'array_planar_regular_orthogonal:NoPlanarOrthotopicalFaces';
                error( errorStruct );
            end

            % ensure cell array for element_pitch_axis
            if ~iscell( element_pitch_axis )
                element_pitch_axis = { element_pitch_axis };
            end

            % ensure equal number of dimensions and sizes
            auxiliary.mustBeEqualSize( faces_ref, element_pitch_axis );

            %--------------------------------------------------------------
            % 2.) create regular orthogonal planar transducer arrays
            %--------------------------------------------------------------
            % create cuboid reference cells
            cells_ref = math.cuboid( element_pitch_axis );

            % constructor of superclass
            objects@transducers.array_planar_regular( faces_ref, cells_ref, N_elements_axis );

        end % function objects = array_planar_regular_orthogonal( faces_ref, element_pitch_axis, N_elements_axis )

	end % methods

end % classdef array_planar_regular_orthogonal < transducers.array_planar_regular