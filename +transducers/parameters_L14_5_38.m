%
% parameters for commercially available linear transducer array
% vendor name: Ultrasonix Medical Corporation
% model name: L14-5/38
%
% author: Martin F. Schiffner
% date: 2019-02-18
% modified: 2019-04-02
%
classdef parameters_L14_5_38 < transducers.parameters_planar

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % methods
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    methods

        %------------------------------------------------------------------
        % constructor
        %------------------------------------------------------------------
        function objects = parameters_L14_5_38

            %--------------------------------------------------------------
            % 1.) model and vendor (Ultrasonix Medical Corporation, L14-5/38)
            %--------------------------------------------------------------
            str_model = 'L14-5/38';
            str_vendor = 'Ultrasonix Medical Corporation';

            %--------------------------------------------------------------
            % 2.) geometrical specifications
            %--------------------------------------------------------------
            N_elements_axis	= [128, 1];                                             % number of physical elements along each coordinate axis (1)
            element_width_axis	= physical_values.meter( [ 279.8, 4000 ] * 1e-6 );	% widths of physical elements along each coordinate axis
            element_kerf_axis	= physical_values.meter( [ 25, 0 ] * 1e-6 );        % kerfs between physical elements along each coordinate axis

            %--------------------------------------------------------------
            % 3.) acoustic lens specifications
            %--------------------------------------------------------------
            apodization_axis = { @(x) 1, @(x) 1 };              % apodization along each coordinate axis
            elevational_focus = physical_values.meter( 16e-3 );	% axial distance of the elevational focus

            %--------------------------------------------------------------
            % 4.) electromechanical specifications
            %--------------------------------------------------------------
            f_center  = physical_values.hertz( 7.5e6 );     % temporal center frequency
            B_frac_lb = 0.65;                               % fractional bandwidth at -6 dB (1)

            %--------------------------------------------------------------
            % 5.) constructor of superclass
            %--------------------------------------------------------------
            objects@transducers.parameters_planar( N_elements_axis, element_width_axis, element_kerf_axis, str_model, str_vendor );

        end % function objects = parameters_L14_5_38

	end % methods

end % classdef parameters_L14_5_38 < transducers.parameters_planar
