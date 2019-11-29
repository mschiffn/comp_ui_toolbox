%
% superclass for all steered plane wave (PW) synthesis settings
%
% author: Martin F. Schiffner
% date: 2019-09-24
% modified: 2019-11-28
%
classdef tx_PW < scattering.sequences.settings.controls.tx_QPW

	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	%% properties
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	properties (SetAccess = private)

        % dependent properties
        index_ref ( 1, : ) double { mustBePositive, mustBeInteger }
        position_ref ( 1, : ) physical_values.length	% reference position

	end % properties

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% methods
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	methods

        %------------------------------------------------------------------
        % constructor
        %------------------------------------------------------------------
        function objects = tx_PW( setup, u_tx_tilde, e_theta )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % superclass ensures class scattering.sequences.setups.setup (scalar)
            % superclass ensures class scattering.sequences.setups.transducers.array_planar_regular_orthogonal

            % ensure class discretizations.signal
            if ~isa( u_tx_tilde, 'discretizations.signal' )
                errorStruct.message = 'u_tx_tilde must be discretizations.signal!';
                errorStruct.identifier = 'tx_PW:NoRegularOrthogonalTransducerArray';
                error( errorStruct );
            end

            % superclass ensures class math.unit_vector
            % superclass ensures equal number of dimensions and sizes

            %--------------------------------------------------------------
            % 2.) create synthesis settings for steered plane waves (PWs)
            %--------------------------------------------------------------
            % constructor of superclass
            objects@scattering.sequences.settings.controls.tx_QPW( setup, u_tx_tilde, e_theta );

            % iterate synthesis settings
            for index_object = 1:numel( objects )

                % check steering direction to find the element that fires first
                indicator = objects( index_object ).e_theta.components( 1:(end - 1) ) >= 0;
                indices_axis_ref = indicator .* ones( 1, setup.xdc_array.N_dimensions ) + ~indicator .* setup.xdc_array.N_elements_axis';
                objects( index_object ).index_ref = forward_index_transform( setup.xdc_array, indices_axis_ref );

                % reference positions
%                 objects( index_object ).position_ref = [ ( -1 ).^( indicator ) .* ( setup.xdc_array.N_elements_axis' - 1 ) .* setup.xdc_array.cell_ref.edge_lengths / 2, 0 ];
                objects( index_object ).position_ref = [ setup.xdc_array.positions_ctr( objects( index_object ).index_ref, : ), 0 ];

            end % for index_object = 1:numel( objects )

        end % function objects = tx_PW( setup, u_tx_tilde, e_theta )

	end % methods

end % classdef tx_PW < scattering.sequences.settings.controls.tx_QPW
