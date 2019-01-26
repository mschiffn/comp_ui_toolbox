%
% class for commercially available linear transducer array
% vendor name: Ultrasonix Medical Corporation
% model name: L14-5/38
%
% author: Martin F. Schiffner
% date: 2017-05-03
% modified: 2019-01-17
%
classdef L14_5_38 < transducer_models.planar_transducer_array

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % methods
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    methods

        %------------------------------------------------------------------
        % constructor
        %------------------------------------------------------------------
        function obj = L14_5_38( varargin )

            %--------------------------------------------------------------
            % optional number of dimensions
            %--------------------------------------------------------------
            % specify two-dimensional transducer array as default
            N_dimensions = 2;

            % process optional argument
            if numel( varargin ) > 0

                % ensure correct number of dimensions
                if ~( ( varargin{ 1 } == 1 ) || ( varargin{ 1 } == 2 ) )
                    errorStruct.message     = 'Number of dimensions N_dimensions must equal either 1 or 2!';
                    errorStruct.identifier	= 'L14_5_38:DimensionMismatch';
                    error( errorStruct );
                end
                N_dimensions = varargin{ 1 };
            end
            % assertion: N_dimensions == 1 || N_dimensions == 2

            %--------------------------------------------------------------
            % name and vendor (Ultrasonix Medical Corporation, L14-5/38)
            %--------------------------------------------------------------
            str_name = 'L14-5/38 (Ultrasonix Medical Corporation)';

            %--------------------------------------------------------------
            % geometrical specifications
            %--------------------------------------------------------------
            N_elements_axis	= [128, 1];                 % number of physical elements along each coordinate axis (1)
            element_width	= [279.8, 4000] * 1e-6;     % widths of physical elements along each coordinate axis (m)
            element_kerf	= [25, 0] * 1e-6;           % kerfs between physical elements along each coordinate axis (m)

            %--------------------------------------------------------------
            % acoustic lens specifications
            %--------------------------------------------------------------
            elevational_focus = 16e-3;                  % axial distance of the elevational focus (m)

            %--------------------------------------------------------------
            % electromechanical specifications
            %--------------------------------------------------------------
            f_center  = 7.5e6;                          % temporal center frequency (Hz)
            B_frac_lb = 0.65;                           % fractional bandwidth at -6 dB (1)

            %--------------------------------------------------------------
            % constructor of superclass
            %--------------------------------------------------------------
            obj@transducer_models.planar_transducer_array( N_elements_axis( 1:N_dimensions ), element_width( 1:N_dimensions ), element_kerf( 1:N_dimensions ), str_name );
        end

	end % methods

end % classdef L14_5_38 < transducer_models.planar_transducer_array
