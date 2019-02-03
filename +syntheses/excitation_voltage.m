%
% superclass for all excitation voltages
%
% author: Martin F. Schiffner
% date: 2017-04-19
% modified: 2019-02-03
%
classdef excitation_voltage < physical_values.signal

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % methods
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	methods

        %------------------------------------------------------------------
        % constructor
        %------------------------------------------------------------------
        function objects = excitation_voltage( u_tx_tilde, f_s )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % ensure u_tx_tilde is a cell array
            if ~iscell( u_tx_tilde )
                u_tx_tilde = { u_tx_tilde };
            end

            % ensure class physical_values.voltage
            for index_signal = 1:numel( u_tx_tilde )

                if ~isa( u_tx_tilde{ index_signal }, 'physical_values.voltage' )
                    errorStruct.message     = sprintf( 'u_tx_tilde{ %d } must be physical_values.voltage!', index_signal );
                    errorStruct.identifier	= 'excitation_voltage:NoVoltage';
                    error( errorStruct );
                end
            end

            %--------------------------------------------------------------
            % 2.) constructor of superclass
            %--------------------------------------------------------------
            objects@physical_values.signal( h_tilde, f_s );

        end % function objects = excitation_voltage( u_tx_tilde, f_s )

    end % methods

end
