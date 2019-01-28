%
% superclass for all pulse-echo measurements
%
% author: Martin F. Schiffner
% date: 2019-01-07
% modified: 2019-01-24
%
classdef measurement

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % properties
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	properties (SetAccess = private)

        % independent properties
        interval_t %( 1, 1 ) physical_values.time_interval           % recording time interval
        interval_f %( 1, 1 ) physical_values.frequency_interval      % frequency interval
        settings_tx %( 1, 1 ) syntheses.setting                      % synthesis settings
        u_rx %( :, : ) physical_values.voltage_phasor               % Fourier coefficients of the recorded RF voltage signals

        % dependent properties
        set_f %( 1, 1 ) frequencies.discrete_frequency_set       % set of relevant discrete frequencies
    end % properties

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % methods
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    methods

        %------------------------------------------------------------------
        % constructor
        %------------------------------------------------------------------
        function objects = measurement( intervals_t, intervals_f, settings_tx, u_rx )

            % check number of arguments
            if nargin ~= 4
                errorStruct.message     = 'The number of arguments must equal four!';
                errorStruct.identifier	= 'measurement:Arguments';
                error( errorStruct );
            end

            % prevent emptyness of the arguments
            mustBeNonempty( intervals_t );
            mustBeNonempty( intervals_f );
            mustBeNonempty( settings_tx );

            % TODO: check size
            % construct column vector of objects
            N_measurements = size( settings_tx, 1 );
            objects = repmat( objects, [ N_measurements, 1 ] );

            % process pulse-echo measurements
            for index_meas = 1:N_measurements

                % set independent properties
                objects( index_meas ).interval_t	= intervals_t( index_meas );
                objects( index_meas ).interval_f	= intervals_f( index_meas );
                objects( index_meas ).settings_tx	= settings_tx( index_meas );
%               objects( index_meas ).u_rx = u_rx( index_meas );

                % determine dependent properties
                objects( index_meas ).set_f = discretization.discrete_frequency_set( objects( index_meas ).interval_f, abs( objects( index_meas ).interval_t ) );
            end           
        end % function objects = measurement( intervals_t, intervals_f, settings_tx, u_rx )

        %------------------------------------------------------------------
        % compute hash value
        %------------------------------------------------------------------
        function str_hash = hash( object )

            % use DataHash function to compute hash value
            str_hash = auxiliary.DataHash( object );

        end % function str_hash = hash( object )

	end % methods

end % classdef measurement
