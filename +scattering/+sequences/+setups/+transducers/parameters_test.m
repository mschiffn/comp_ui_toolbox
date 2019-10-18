%
% parameters for commercially available linear transducer array
% vendor name: Ultrasonix Medical Corporation
% model name: L14-5/38
%
% author: Martin F. Schiffner
% date: 2019-02-18
% modified: 2019-04-11
%
classdef parameters_test < transducers.parameters_planar

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % methods
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    methods

        %------------------------------------------------------------------
        % constructor
        %------------------------------------------------------------------
        function objects = parameters_test

            %--------------------------------------------------------------
            % 1.) model and vendor
            %--------------------------------------------------------------
            str_model = 'Test';
            str_vendor = 'Test Corporation';

            %--------------------------------------------------------------
            % 2.) geometrical specifications
            %--------------------------------------------------------------
            N_elements_axis	= [1, 1];                                               % number of physical elements along each coordinate axis (1)
            element_width_axis	= physical_values.meter( [ 2, 2 ] * 1e-2 );         % widths of physical elements along each coordinate axis
            element_kerf_axis	= physical_values.meter( [ 0, 0 ] * 1e-6 );         % kerfs between physical elements along each coordinate axis

            %--------------------------------------------------------------
            % 3.) acoustic lens specifications
            %--------------------------------------------------------------
            apodization = @( r ) cos( pi * r( :, 1 ) / 2 );                     % apodization along each coordinate axis
            axial_focus_axis = physical_values.meter( [ 4, 4 ] * 1e-3 );        % axial distance of the elevational focus (TODO: no focus possible using inf?)

            %--------------------------------------------------------------
            % 4.) electromechanical specifications
            %--------------------------------------------------------------
            f_center  = physical_values.hertz( 7.5e6 );     % temporal center frequency
            B_frac_lb = 0.65;                               % fractional bandwidth at -6 dB (1)

            %--------------------------------------------------------------
            % 5.) constructor of superclass
            %--------------------------------------------------------------
            objects@transducers.parameters_planar( N_elements_axis, element_width_axis, element_kerf_axis, apodization, axial_focus_axis, str_model, str_vendor );

        end % function objects = parameters_test

	end % methods

end % classdef parameters_test < transducers.parameters_planar
