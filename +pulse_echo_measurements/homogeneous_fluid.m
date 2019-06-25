%
% superclass for all lossy homogeneous fluids
%
% author: Martin F. Schiffner
% date: 2018-06-01
% modified: 2019-06-12
%
classdef homogeneous_fluid

	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	%% properties
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	properties (SetAccess = private)

        % independent properties
        rho_0 ( 1, 1 ) physical_values.mass_density = physical_values.kilogram_per_cubicmeter( 1000 )	% unperturbed mass density
        absorption_model ( 1, 1 ) absorption_models.absorption_model = absorption_models.time_causal( 0, 0.5, 1, physical_values.meter_per_second( 1540 ), physical_values.hertz( 4e6 ), 1 )	% absorption model for the lossy homogeneous fluid
% TODO: c_avg vs c_ref? group velocity?
        c_avg ( 1, 1 ) physical_values.velocity = physical_values.meter_per_second( 1540 );	% average small-signal sound speed

    end % properties

	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	%% methods
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	methods

        %------------------------------------------------------------------
        % constructor
        %------------------------------------------------------------------
        function objects = homogeneous_fluid( rho_0, absorption_models, c_avg )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % return if no argument
            if nargin == 0
                return;
            end

            % property validation function ensures class physical_values.mass_density for rho_0

            % property validation function ensures class absorption_models.absorption_model for absorption_models

            % property validation function ensures class physical_values.meter_per_second for c_avg

            % ensure equal number of dimensions and sizes
            auxiliary.mustBeEqualSize( rho_0, absorption_models, c_avg );

            %--------------------------------------------------------------
            % 2.) create lossy homogeneous fluids
            %--------------------------------------------------------------
            % repeat default lossy homogeneous fluid
            objects = repmat( objects, size( rho_0 ) );

            % iterate lossy homogeneous fluids
            for index_object = 1:numel( objects )

                %----------------------------------------------------------
                % a) set independent properties
                %----------------------------------------------------------
                objects( index_object ).rho_0 = rho_0( index_object );
                objects( index_object ).absorption_model = absorption_models( index_object );
                objects( index_object ).c_avg = c_avg( index_object );

            end % for index_object = 1:numel( objects )

        end % function objects = homogeneous_fluid( rho_0, absorption_models )

    end % methods

end % classdef homogeneous_fluid
