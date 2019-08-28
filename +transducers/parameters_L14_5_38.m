%
% parameters for commercially available linear transducer array
% vendor name: Ultrasonix Medical Corporation
% model name: L14-5/38
%
% author: Martin F. Schiffner
% date: 2019-02-18
% modified: 2019-06-04
%
classdef parameters_L14_5_38

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% properties
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	properties (SetAccess = private)

            %--------------------------------------------------------------
            % 1.) model and vendor (Ultrasonix Medical Corporation, L14-5/38)
            %--------------------------------------------------------------
            str_model = 'L14-5/38';
            str_vendor = 'Ultrasonix Medical Corporation';

            %--------------------------------------------------------------
            % 2.) geometrical specifications
            %--------------------------------------------------------------
            N_elements_axis = [ 128, 1 ];                                           % number of physical elements along each coordinate axis (1)
            element_width_axis	= physical_values.meter( [ 279.8, 4000 ] * 1e-6 );	% widths of physical elements along each coordinate axis
            element_kerf_axis	= physical_values.meter( [ 25, 0 ] * 1e-6 );        % kerfs between physical elements along each coordinate axis

            %--------------------------------------------------------------
            % 3.) acoustic lens specifications
            %--------------------------------------------------------------
            apodization = @transducers.apodization.uniform;                         % apodization along each coordinate axis
            axial_focus_axis = physical_values.meter( [ Inf, 16e-3 ] );             % axial distances of the lateral foci
            absorption_model = absorption_models.none( physical_values.meter_per_second( 1540 ) );	% absorption model for the acoustic lens

            %--------------------------------------------------------------
            % 4.) electromechanical specifications
            %--------------------------------------------------------------
            f_center  = physical_values.hertz( 7.5e6 );     % temporal center frequency
            B_frac_lb = 0.65;                               % fractional bandwidth at -6 dB (1)

    end % properties

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% methods
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    methods

        %------------------------------------------------------------------
        % constructor
        %------------------------------------------------------------------
        function objects = parameters_L14_5_38

            %

            %--------------------------------------------------------------
            % 5.) constructor of superclass
            %--------------------------------------------------------------
%             objects@transducers.parameters_planar_regular_orthogonal( N_elements_axis, element_width_axis, element_kerf_axis, apodization, axial_focus_axis, str_model, str_vendor );

        end % function objects = parameters_L14_5_38

	end % methods

end % classdef parameters_L14_5_38 < transducers.parameters_planar_orthotope
