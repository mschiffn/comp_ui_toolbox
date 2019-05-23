%
% superclass for all synthesis settings for quasi-(d-1)-spherical waves
%
% author: Martin F. Schiffner
% date: 2019-01-21
% modified: 2019-05-21
%
classdef setting_tx_QSW < controls.setting_tx

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% properties
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	properties (SetAccess = private)

        % independent properties
        positions_src ( 1, : ) physical_values.length	% positions of the virtual sources
        angles ( 1, : ) double                          % aperture angles (rad); 0 < angles < pi

    end % properties

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% methods
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    methods

        %------------------------------------------------------------------
        % constructor
        %------------------------------------------------------------------
        function objects = setting_tx_QSW( setup, excitation_voltages_common, positions_src, angles )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % ensure class pulse_echo_measurements.setup
            if ~( isa( setup, 'pulse_echo_measurements.setup' ) && isscalar( setup ) )
                errorStruct.message = 'setup must be a single pulse_echo_measurements.setup!';
                errorStruct.identifier = 'setting_tx_QSW:NoSetup';
                error( errorStruct );
            end

            % excitation_voltages_common will be checked in superclass

            % ensure class physical_values.length
            if ~isa( positions_src, 'physical_values.length' )
                errorStruct.message = 'positions_src must be physical_values.length!';
                errorStruct.identifier = 'setting_tx_QSW:NoLength';
                error( errorStruct );
            end

            % ensure correct number of dimensions for positions_src
            if size( positions_src, 2 ) ~= setup.FOV.N_dimensions
                errorStruct.message = 'The second dimension of positions_src must match the number of dimensions!';
                errorStruct.identifier = 'setting_tx_QSW:DimensionMismatch';
                error( errorStruct );
            end

            % number of lateral dimensions
            N_dimensions_lateral = setup.FOV.N_dimensions - 1;

            % ensure correct number of dimensions for angles
            if size( angles, 2 ) ~= N_dimensions_lateral
                errorStruct.message = 'The second dimension of angles must match the number of dimensions minus unity!';
                errorStruct.identifier = 'setting_tx_QSW:DimensionMismatch';
                error( errorStruct );
            end

            % ensure positive aperture angles
            mustBePositive( angles );
            mustBeLessThan( angles, pi );

            %--------------------------------------------------------------
            % 2.) compute synthesis settings for QSWs
            %--------------------------------------------------------------
            % number of sequential syntheses
            N_objects = numel( excitation_voltages_common );

            % allocate cell arrays to store synthesis settings
            indices_active = cell( size( excitation_voltages_common ) );
            impulse_responses = cell( size( excitation_voltages_common ) );
            excitation_voltages = cell( size( excitation_voltages_common ) );

            % iterate synthesis settings
            for index_object = 1:N_objects

                %----------------------------------------------------------
                % a) determine active array elements
                %----------------------------------------------------------
                vectors_src_ctr = centers( setup.xdc_array ) - repmat( positions_src( index_object, : ), [ setup.xdc_array.N_elements, 1 ] );
                distances_src_ctr = vecnorm( vectors_src_ctr, 2, 2 );
                indicator_distance = double( distances_src_ctr ) >= eps;
                indicator_active = false( size( setup.xdc_array.aperture ) );
                indicator_active( ~indicator_distance ) = true;
                indicator_active( indicator_distance ) = all( asin( abs( vectors_src_ctr( indicator_distance, 1:N_dimensions_lateral ) ./ distances_src_ctr( indicator_distance ) ) ) <= angles( index_object, : ) / 2, 2 );
                indices_active{ index_object } = find( indicator_active );

                %----------------------------------------------------------
                % b) impulse responses are delays
                %----------------------------------------------------------
                % compute time delays for each virtual source
                time_delays_act = distances_src_ctr( indicator_active ) / setup.c_avg;
                time_delays_act = time_delays_act - min( time_delays_act );

                % specify impulse responses
                indices_q = round( time_delays_act / setup.T_clk );
                impulse_responses{ index_object } = discretizations.delta_matrix( indices_q, setup.T_clk, physical_values.meter_per_volt_second( ones( size( indices_q ) ) ) );

                %----------------------------------------------------------
                % c) identical excitation voltages for all array elements
                %----------------------------------------------------------
                excitation_voltages{ index_object } = excitation_voltages_common( index_object );

            end % for index_object = 1:N_objects

            %--------------------------------------------------------------
            % 3.) constructor of superclass
            %--------------------------------------------------------------
            objects@controls.setting_tx( indices_active, impulse_responses, excitation_voltages );

            %--------------------------------------------------------------
            % 4.) set independent properties
            %--------------------------------------------------------------
            for index_object = 1:N_objects

                objects( index_object ).positions_src = positions_src( index_object, : );
                objects( index_object ).angles = angles( index_object, : );

            end

        end % function objects = setting_tx_QSW( setup, excitation_voltages_common, positions_src, angles )

	end % methods

end % classdef setting_tx_QSW < controls.setting_tx
