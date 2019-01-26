%
% superclass for all pulse-echo measurements using the conventional ADC
%
% author: Martin F. Schiffner
% date: 2019-01-22
% modified: 2019-01-22
%
classdef measurement_ADC < pulse_echo_measurements.measurement

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % properties
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	properties (SetAccess = private)

        % independent properties
        f_s ( 1, 1 ) double { mustBePositive, mustBeFinite, mustBeNonempty } = 20e6     % sampling rate

        % dependent properties
        set_t %( 1, 1 )                             % set of time samples
    end % properties

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % methods
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    methods

        %------------------------------------------------------------------
        % constructor
        %------------------------------------------------------------------
        function obj = measurement_ADC( intervals_t, intervals_f, settings_tx, u_rx, f_s )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % check number of arguments
            if nargin ~= 5
                errorStruct.message     = 'The number of arguments must equal four!';
                errorStruct.identifier	= 'measurement_ADC:Arguments';
                error( errorStruct );
            end

            %--------------------------------------------------------------
            % 2.) quantize recording intervals
            %--------------------------------------------------------------
            intervals_t_quantized = quantize( intervals_t, 1 ./ f_s );

            %--------------------------------------------------------------
            % 3.) constructor of the superclass
            %--------------------------------------------------------------
            obj@pulse_echo_measurements.measurement( intervals_t_quantized, intervals_f, settings_tx, u_rx )

            % determine dependent properties
        end
	end % methods

end % classdef measurement_ADC
