%
% superclass for all scattering operators
%
% author: Martin F. Schiffner
% date: 2019-02-14
% modified: 2019-03-16
%
classdef operator

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % properties
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	properties (SetAccess = private)

        % independent properties
        sequence %( 1, 1 ) pulse_echo_measurements.sequence         % pulse-echo measurement sequence
        options ( 1, 1 ) scattering.options                         % scattering operator options

        % dependent properties
        discretization ( 1, 1 ) discretizations.spatiospectral      % results of the spatiospectral discretization
        p_incident % ( :, : ) syntheses.pressure_incident           % incident acoustic pressure field
        p_incident_grad                                             % spatial gradient of the incident acoustic pressure field

    end % properties

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % methods
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    methods

        %------------------------------------------------------------------
        % constructor
        %------------------------------------------------------------------
        function object = operator( sequence, options )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % ensure class pulse_echo_measurements.sequence (scalar)
            if ~( isa( sequence, 'pulse_echo_measurements.sequence' ) && isscalar( sequence ) )
                errorStruct.message     = 'sequence must be a single pulse_echo_measurements.sequence!';
                errorStruct.identifier	= 'operator:NoScalarSequence';
                error( errorStruct );
            end

            % ensure class scattering.options (scalar)
            if ~( isa( options, 'scattering.options' ) && isscalar( options ) )
                errorStruct.message     = 'options must be a single scattering.options!';
                errorStruct.identifier	= 'operator:NoScalarOptions';
                error( errorStruct );
            end

            %--------------------------------------------------------------
            % 2.) set independent properties
            %--------------------------------------------------------------
            object.sequence = sequence;
            object.options = options;

            % TODO: check for identical recording time intervals / identical frequency intervals
            % TODO: which elements are active? -> D_full required?
            % TODO: find active elements and compute impulse responses
            % check for identical frequency axes identical?

            %--------------------------------------------------------------
            % 3.) spatiospectral discretization of the sequence
            %--------------------------------------------------------------
            object.discretization = discretize( object.sequence, object.options.discretization );

            %--------------------------------------------------------------
            % 4.) compute incident acoustic fields (unique frequencies)
            %--------------------------------------------------------------
            object.p_incident = syntheses.pressure_incident( object.sequence.setup, object.discretization );

            % create pulse-echo measurements
            % TODO: check method to determine Fourier coefficients
%             f_s = physical_values.frequency( 20e6 );
%             object.measurements = pulse_echo_measurements.measurement_ADC( intervals_t, intervals_f, settings_tx_quantized, [], repmat( f_s, size( intervals_t ) ) );

        end % function object = operator( sequence, options )

    end % methods

end % classdef operator
