%
% class for commercially available linear transducer array
% vendor name: Verasonics, Inc. (Kirkland, WA, USA)
% model name: L11-5v
%
% author: Martin F. Schiffner
% date: 2019-09-03
% modified: 2019-10-17
%
classdef L11_5v < scattering.sequences.setups.transducers.array_planar_regular_orthogonal

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % methods
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    methods

        %------------------------------------------------------------------
        % constructor
        %------------------------------------------------------------------
        function object = L11_5v( varargin )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % ensure nonempty N_dimensions
            if nargin >= 1 && ~isempty( varargin{ 1 } )
                N_dimensions = varargin{ 1 };
            else
                % specify two-dimensional transducer array as default
                N_dimensions = 2;
            end

            % ensure correct number of dimensions
            if ~( ( N_dimensions == 1 ) || ( N_dimensions == 2 ) )
                errorStruct.message = 'Number of dimensions must equal either 1 or 2!';
                errorStruct.identifier = 'L11_5v:InvalidNumberOfDimensions';
                error( errorStruct );
            end

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % project parameters to current number of dimensions
            parameters = scattering.sequences.setups.transducers.parameters_L11_5v;
%             parameters = project( scattering.sequences.setups.transducers.parameters_L11_5v, N_dimensions );

            % create reference face
            lens = scattering.sequences.setups.transducers.lens( parameters.axial_focus_axis, parameters.absorption_model );
            intervals = num2cell( math.interval( physical_values.meter( zeros( 1, N_dimensions ) ), parameters.element_width_axis ) );
            face_ref = scattering.sequences.setups.transducers.face_planar_orthotope( parameters.apodization, lens, intervals{ : } );

            element_pitch_axis = parameters.element_width_axis + parameters.element_kerf_axis;

            %--------------------------------------------------------------
            % 2.) constructor of superclass
            %--------------------------------------------------------------
            object@scattering.sequences.setups.transducers.array_planar_regular_orthogonal( face_ref, element_pitch_axis, parameters.N_elements_axis );

        end % function object = L11_5v( varargin )

	end % methods

end % classdef L11_5v < scattering.sequences.setups.transducers.array_planar_regular_orthogonal
