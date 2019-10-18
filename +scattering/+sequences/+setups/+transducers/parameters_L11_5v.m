%
% parameters for commercially available linear transducer array
% vendor name: Verasonics, Inc. (Kirkland, WA, USA)
% model name: L11-5v
%
% author: Martin F. Schiffner
% date: 2019-09-03
% modified: 2019-10-17
%
classdef parameters_L11_5v

	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	%% properties
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	properties (SetAccess = private)

        %------------------------------------------------------------------
        % 1.) model and vendor (Verasonics, Inc., L11-5v)
        %------------------------------------------------------------------
        str_model = 'L11-5v';
        str_vendor = 'Verasonics, Inc.';

        %------------------------------------------------------------------
        % 2.) geometrical specifications
        %------------------------------------------------------------------
        N_elements_axis = [ 128, 1 ];                                           % number of physical elements along each coordinate axis (1)
        element_width_axis = physical_values.meter( [ 270, 5000 ] * 1e-6 );     % widths of physical elements along each coordinate axis
        element_kerf_axis  = physical_values.meter( [ 30, 0 ] * 1e-6 );         % kerfs between physical elements along each coordinate axis

        %------------------------------------------------------------------
        % 3.) acoustic lens specifications
        %------------------------------------------------------------------
        apodization = @scattering.sequences.setups.transducers.apodization.uniform; % apodization along each coordinate axis
        axial_focus_axis = physical_values.meter( [ Inf, 18e-3 ] );                 % axial distances of the lateral foci
        absorption_model = scattering.sequences.setups.materials.absorption_models.none( physical_values.meter_per_second( 1540 ) ); % absorption model for the acoustic lens

        %------------------------------------------------------------------
        % 4.) electromechanical specifications
        %------------------------------------------------------------------
        f_center  = physical_values.hertz( 5.208e6 );	% temporal center frequency
        B_frac_lb = 0.67;                               % fractional bandwidth at -6 dB (1)

    end % properties

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% methods
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    methods

        %------------------------------------------------------------------
        % constructor
        %------------------------------------------------------------------
        function objects = parameters_L11_5v

            %

            %--------------------------------------------------------------
            % 5.) constructor of superclass
            %--------------------------------------------------------------
%             objects@scattering.sequences.setups.transducers.parameters_planar_regular_orthogonal( N_elements_axis, element_width_axis, element_kerf_axis, apodization, axial_focus_axis, str_model, str_vendor );

        end % function objects = parameters_L11_5v

	end % methods

end % classdef parameters_L11_5v
