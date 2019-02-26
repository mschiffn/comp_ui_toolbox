%
% superclass for all synthesis settings
%
% author: Martin F. Schiffner
% date: 2019-02-25
% modified: 2019-02-26
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

	end % methods

end % classdef setting_tx < controls.setting
