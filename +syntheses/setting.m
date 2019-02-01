%
% superclass for all synthesis settings
%
% author: Martin F. Schiffner
% date: 2019-01-07
% modified: 2019-02-01
%
classdef setting

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % properties
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	properties (SetAccess = private)

        % independent properties
        indices_active ( 1, : ) double { mustBeInteger, mustBeFinite }      % indices of active array elements (1)
        apodization_weights ( 1, : ) physical_values.apodization_weight     % apodization weights
        time_delays ( 1, : ) physical_values.time                           % time delays
        excitation_voltages ( 1, : ) syntheses.excitation_voltage           % excitation voltages
    end % properties

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % methods
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    methods

        %------------------------------------------------------------------
        % constructor
        %------------------------------------------------------------------
        function objects = setting( indices_active, apodization_weights, time_delays, excitation_voltages )

            % return if no argument
            if nargin == 0
                return;
            end

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % check type of arguments
            if ~iscell( indices_active ) || ~iscell( apodization_weights ) || ~iscell( time_delays ) || ~iscell( excitation_voltages )
                errorStruct.message     = 'indices_active, apodization_weights, time_delays, and excitation_voltages must be cell arrays!';
                errorStruct.identifier	= 'setting:NoCellArrays';
                error( errorStruct );
            end
            % assertion: all arguments are cell arrays

            % ensure equal sizes of cell arrays
            if sum( size( indices_active ) ~= size( apodization_weights ) ) || sum( size( indices_active ) ~= size( time_delays ) ) || sum( size( indices_active ) ~= size( excitation_voltages ) )
                errorStruct.message     = 'The sizes of indices_active, apodization_weights, time_delays, and excitation_voltages must match!';
                errorStruct.identifier	= 'setting:SizeMismatch';
                error( errorStruct );
            end
            % assertion: all cell arrays have equal sizes

            %--------------------------------------------------------------
            % 2.) create synthesis settings
            %--------------------------------------------------------------
            % construct column vector of objects
            N_objects = numel( indices_active );
            objects = repmat( objects, [ N_objects, 1 ] );

            % set independent properties
            for index_object = 1:N_objects

                % ensure equal sizes of cell array contents
                if sum( size( indices_active{ index_object } ) ~= size( apodization_weights{ index_object } ) ) || sum( size( indices_active{ index_object } ) ~= size( time_delays{ index_object } ) ) || sum( size( indices_active{ index_object } ) ~= size( excitation_voltages{ index_object } ) )
                    errorStruct.message     = sprintf( 'The sizes of indices_active{ %d }, apodization_weights{ %d }, time_delays{ %d }, and excitation_voltages{ %d } must match!', index_object, index_object, index_object, index_object );
                    errorStruct.identifier	= 'setting:SizeMismatch';
                    error( errorStruct );
                end
                % assertion: all cell array contents have equal sizes

                % ensure row vectors
                if ~isrow( indices_active{ index_object } )
                    errorStruct.message     = sprintf( 'The contents of indices_active{ %d }, apodization_weights{ %d }, time_delays{ %d }, and excitation_voltages{ %d } must be row vectors!', index_object, index_object, index_object, index_object );
                    errorStruct.identifier	= 'setting:SizeMismatch';
                    error( errorStruct );
                end

                % set independent properties
                objects( index_object ).indices_active = indices_active{ index_object }( 1, : );
                objects( index_object ).apodization_weights = apodization_weights{ index_object }( 1, : );
                objects( index_object ).time_delays = time_delays{ index_object }( 1, : );
                objects( index_object ).excitation_voltages = excitation_voltages{ index_object }( 1, : );

            end % for index_object = 1:N_objects

            % reshape column vector to size of the cell arrays
            objects = reshape( objects, size( indices_active ) );

        end % function objects = setting( indices_active, apodization_weights, time_delays, excitation_voltages )

        %------------------------------------------------------------------
        % quantization
        %------------------------------------------------------------------
        function objects = quantize( objects, T_clk )

            % TODO: quantize apodization weights

            % quantize time delays
            N_objects = size( objects, 1 );
            for index_object = 1:N_objects
                % quantize time delays
                objects( index_object ).time_delays = quantize( objects( index_object ).time_delays, T_clk );
            end
        end % function objects = quantize( objects, T_clk )

	end % methods

end % classdef setting
