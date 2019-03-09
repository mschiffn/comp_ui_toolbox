%
% superclass for all synthesis settings
%
% author: Martin F. Schiffner
% date: 2019-02-25
% modified: 2019-03-05
%
classdef setting_tx < controls.setting

	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	% properties
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	properties (SetAccess = private)

        % independent properties
        excitation_voltages ( 1, : ) syntheses.excitation_voltage	% excitation voltages

    end % properties

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % methods
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	methods

        %------------------------------------------------------------------
        % constructor
        %------------------------------------------------------------------
        function objects = setting_tx( indices_active, impulse_responses, excitation_voltages )

            %--------------------------------------------------------------
            % 1.) constructor of superclass
            %--------------------------------------------------------------
            if nargin == 0
                indices_active = 1;
                impulse_responses = physical_values.impulse_response( discretizations.set_discrete_time_regular( 0, 0, physical_values.time(1) ), physical_values.physical_value(1) );
                excitation_voltages = syntheses.excitation_voltage( discretizations.set_discrete_time_regular( 0, 0, physical_values.time(1) ), physical_values.voltage(1) );
            end

            objects@controls.setting( indices_active, impulse_responses );

            %--------------------------------------------------------------
            % 2.) check arguments
            %--------------------------------------------------------------
            % ensure cell array
            if ~iscell( excitation_voltages )
                excitation_voltages = { excitation_voltages };
            end

            % ensure equal number of dimensions and sizes
            auxiliary.mustBeEqualSize( objects, excitation_voltages );

            %--------------------------------------------------------------
            % 3.) create synthesis settings
            %--------------------------------------------------------------
            % set independent properties
            for index_object = 1:numel( objects )

                objects( index_object ).excitation_voltages = excitation_voltages{ index_object };

            end % for index_object = 1:numel( indices_active )

        end % function objects = setting_tx( indices_active, impulse_responses, excitation_voltages )

        %------------------------------------------------------------------
        % spectral discretization
        %------------------------------------------------------------------
        function objects_out = discretize( setting_tx, intervals_t, intervals_f )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % ensure equal number of dimensions and sizes
            auxiliary.mustBeEqualSize( intervals_t, intervals_f );

            %--------------------------------------------------------------
            % 2.) compute transfer functions and excitation voltages
            %--------------------------------------------------------------
            % initialize cell arrays
            transfer_functions = cell( size( intervals_t ) );
            excitation_voltages = cell( size( intervals_t ) );

            % iterate intervals
            for index_object = 1:numel( intervals_t )

                transfer_functions{ index_object } = fourier_transform( setting_tx.impulse_responses, intervals_t( index_object ), intervals_f( index_object ) );
                excitation_voltages{ index_object } = fourier_coefficients( setting_tx.excitation_voltages, intervals_t( index_object ), intervals_f( index_object ) );

            end % for index_object = 1:numel( intervals_t )

            %--------------------------------------------------------------
            % 3.) create spectral discretizations of the recording settings
            %--------------------------------------------------------------
            objects_out = discretizations.spectral_points_tx( transfer_functions, excitation_voltages );

        end % function objects_out = discretize( setting_tx, intervals_t, intervals_f )

	end % methods

end % classdef setting_tx < controls.setting
