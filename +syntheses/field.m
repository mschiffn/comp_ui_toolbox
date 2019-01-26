%
% superclass for all fields
%
% author: Martin F. Schiffner
% date: 2019-01-22
% modified: 2019-01-24
%
classdef field

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % properties
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	properties

        % independent properties
        values % phasors of the acoustic value for each grid point in the FOV

        % dependent properties
        size_bytes % ( 1, 1 ) physical_values.memory % memory consumption (B)
    end % properties

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % methods
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    methods

        %------------------------------------------------------------------
        % constructor
        %------------------------------------------------------------------
        function objects = field( setup, measurements )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % return if no input argument
            if nargin == 0
                return;
            end

            % check if pulse-echo measurement setup is discretized
            if ~isa( setup, 'pulse_echo_measurements.setup' ) || numel( setup ) ~= 1 || isempty( setup.D_ctr )
                errorStruct.message     = 'setup must be a single discretized pulse_echo_measurements.setup!';
                errorStruct.identifier	= 'field:NoDiscretizedSetup';
                error( errorStruct );
            end

            % check if measurements are valid
            if ~isa( measurements, 'pulse_echo_measurements.measurement' )
                errorStruct.message     = 'measurements must be pulse_echo_measurements.measurement!';
                errorStruct.identifier	= 'pressure_incident:WrongObjects';
                error( errorStruct );
            end

            %--------------------------------------------------------------
            % 2.) construct objects
            %--------------------------------------------------------------
            % construct column vector of objects
            N_objects = size( measurements, 1 );
            objects = repmat( objects, [ N_objects, 1 ] );

            %--------------------------------------------------------------
            % 3.) initialize objects with zeros / compute memory consumption
            %--------------------------------------------------------------
            for index_object = 1:N_objects

                % number of discrete frequencies
                N_samples_f = numel( [ measurements( index_object ).set_f.F_BP.value ] );
                N_points = setup.FOV.grid.N_points;

                % initialize field values with zeros
                objects( index_object ).values = cell( 1, N_samples_f );
                for index_f = 1:N_samples_f
                    objects( index_object ).values{ index_f } = zeros( setup.FOV.grid.N_points_axis(2), setup.FOV.grid.N_points_axis(1) );
                end

                % compute memory consumption
                objects( index_object ).size_bytes = physical_values.memory( N_samples_f * N_points * 16 );
            end
        end

        %------------------------------------------------------------------
        % show (overload display function)
        %------------------------------------------------------------------
        function hdl = show( objects )

            N_objects = size( objects, 1 );
            hdl = zeros( N_objects, 1 );
            for index_object = 1:N_objects

                N_samples_f = numel( objects( index_object ).values );
                index_ctr = round( N_samples_f / 2 );

                hdl( index_object ) = figure( index_object );
                subplot( 2, 3, 1);
                imagesc( abs( objects( index_object ).values{ 1 } ) );
                subplot( 2, 3, 2);
                imagesc( abs( objects( index_object ).values{ index_ctr } ) );
                subplot( 2, 3, 3);
                imagesc( abs( objects( index_object ).values{ end } ) );
                subplot( 2, 3, 4);
                imagesc( angle( objects( index_object ).values{ 1 } ) );
                subplot( 2, 3, 5);
                imagesc( angle( objects( index_object ).values{ index_ctr } ) );
                subplot( 2, 3, 6);
                imagesc( angle( objects( index_object ).values{ end } ) );
            end
        end % function hdl = show( objects )

	end % methods

end % classdef field
