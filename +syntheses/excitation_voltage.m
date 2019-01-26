%
% superclass for all excitation voltages
%
% author: Martin F. Schiffner
% date: 2017-04-19
% modified: 2019-01-23
%
classdef excitation_voltage

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % properties
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	properties (SetAccess = private)

        % independent properties
        u_tx_tilde %( 1, : ) physical_values.voltage     % samples of the excitation voltage
        f_s %( 1, 1 ) physical_values.frequency          % sampling rate

        % dependent properties
        T_duration
    end

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % methods
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	methods

        %------------------------------------------------------------------
        % constructor
        %------------------------------------------------------------------
        function objects = excitation_voltage( u_tx_tilde, f_s )

            % check u_tx_tilde
            if ~iscell( u_tx_tilde )
                errorStruct.message     = 'u_tx_tilde must be a cell array!';
                errorStruct.identifier	= 'excitation_voltage:NoCellArray';
                error( errorStruct );
            end

            % ensure equal dimensions
            if sum( size( u_tx_tilde ) ~= size( f_s ) )
                errorStruct.message     = 'The dimensions of u_tx_tilde and f_s must match!';
                errorStruct.identifier	= 'excitation_voltage:DimensionMismatch';
                error( errorStruct );
            end

            % construct column vector of objects
            N_objects = numel( u_tx_tilde );
            objects = repmat( objects, [ N_objects, 1 ] );

            % set independent properties
            for index_object = 1:N_objects
                objects( index_object ).u_tx_tilde = u_tx_tilde{ index_object }( : )';
                objects( index_object ).f_s = f_s( index_object );
            end

            % reshape to dimensions of the argument
            objects = reshape( objects, size( u_tx_tilde ) );
        end
    end % methods
end
